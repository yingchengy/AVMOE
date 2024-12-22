import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from einops import rearrange, repeat
import os

import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

from htsat import HTSAT_Swin_Transformer
import esc_config as esc_config
from utils import do_mixup, get_mix_lambda, do_mixup_label


class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = opt.is_multimodal
        self.opt = opt
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        self.fc = nn.Linear(linear_in, linear_out)

        d_model = linear_out // 2
        self.fc_affine_audio_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_video_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_bottleneck = nn.Linear(linear_out, d_model)
        self.fc_affine_video_2 = nn.Linear(linear_out, d_model)
        self.fc_affine_audio_2 = nn.Linear(linear_out, d_model)
        self.fc_affine_v_s_att = nn.Linear(d_model, 1)
        self.fc_tanh = nn.Tanh()
        self.fc_softmax = nn.Softmax(dim=-1)
        self.fc_affine_v_c_att = nn.Linear(d_model, linear_out)

        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor

            self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
            self.gate_av = nn.Parameter(torch.zeros(1))

            ### <------

            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.num_head = 4
            self.head_dropout = 0.2
            if self.opt.is_self_attention:
                self.self_attention = MultiheadAttention(input_dim, num_heads=self.num_head, dropout=self.head_dropout)

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)

            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token.permute(0, 2, 1).unsqueeze(-1)

        spatial_att_maps = None
        if self.adapter_kind == "bottleneck" and self.is_multimodal:

            ### -------> high dim att
            rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))

            att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))

            rep_token = rep_token + rep_token_res

            att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))

            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

            x = x + self.gate_av * x_res.contiguous()

            # ============================== Channel Attention ====================================
            audio = vis_token.mean(dim=2).squeeze(-1)  # [B*10, dim]
            audio_query_1 = F.relu(self.fc_affine_audio_1(audio)).unsqueeze(-2)
            video_query_1 = F.relu(
                self.fc_affine_video_1(x.squeeze(-1).permute(0, 2, 1)))  # [*, grid ** 2, width]
            audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)  # [*, width]
            audio_video_query = F.relu(self.fc_affine_bottleneck(audio_video_query_raw))
            channel_att_maps = self.fc_affine_v_c_att(audio_video_query).sigmoid().reshape(x.size(0), 1, -1)
            c_att_visual_feat = (x.squeeze(-1).permute(0, 2, 1) * (channel_att_maps + 1))  # [B*10, 36, 768]

            # ============================== Spatial Attention =====================================
            # channel attended visual feature: [batch * 10, 36, v_dim]
            c_att_visual_query = F.relu(self.fc_affine_video_2(c_att_visual_feat))
            audio_query_2 = F.relu(self.fc_affine_audio_2(audio)).unsqueeze(-2)
            audio_video_query_2 = c_att_visual_query * audio_query_2
            spatial_att_maps_tmp = self.fc_affine_v_s_att(audio_video_query_2)
            spatial_att_maps_sigmoid = spatial_att_maps_tmp.transpose(2, 1).sigmoid()
            spatial_att_maps_sigmoid = spatial_att_maps_sigmoid.transpose(2, 1)
            spatial_att_maps = self.fc_softmax(self.fc_tanh(spatial_att_maps_tmp).transpose(2, 1))
            c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)

            alpha, beta = 0.3, 0.05
            x = x.squeeze(-1).permute(0, 2, 1) * (
                    alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
            x = x.permute(0, 2, 1).unsqueeze(-1)

            # =======================================================================================

            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            ## <----

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":

            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)


        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C'))
                output = rearrange(output, 'N L C -> N C L')

        if self.opt.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.gate is not None:
            output = self.gate * output

        return output, spatial_att_maps


class ExpertAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, is_multimodal=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = is_multimodal
        self.opt = opt
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor
            self.my_tokens = nn.Parameter(torch.rand((self.opt.num_tokens, input_dim)))

            self.gate_av = nn.Parameter(torch.zeros(1))
            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            self.num_head = 4
            self.head_dropout = 0.2
            if self.opt.is_self_attention:
                self.self_attention = MultiheadAttention(input_dim, num_heads=self.num_head, dropout=self.head_dropout)

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)

            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            # self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None):
        if self.adapter_kind == "bottleneck" and self.is_multimodal:
            ### -------> high dim att
            rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
            att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))
            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))
            rep_token = rep_token + rep_token_res

            att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))

            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

            x = x + self.gate_av * x_res.contiguous()
            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":
            if self.opt.is_self_attention:
                x = x.squeeze(-1).permute(0,2,1)

                x, x_weights = self.self_attention(x, x, x)
                x = x.permute(0, 2, 1).unsqueeze(-1)
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C'))
                output = rearrange(output, 'N L C -> N C L')

        if self.opt.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.gate is not None:
            output = self.gate * output
        return output


class MoEAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, adapter_kind, dim_list, layer_idx, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.opt = opt
        self.use_bn = use_bn
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
        self.fc = nn.Linear(linear_in, linear_out)
        self.num_multimodal_experts = self.opt.num_multimodal_experts
        self.num_singlemodal_experts = self.opt.num_singlemodal_experts
        self.multimodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
                          layer_idx, reduction_factor, opt, use_bn,
                          use_gate, is_multimodal=True)
            for _ in range(self.num_multimodal_experts)
        ])
        self.singlemodal_experts = nn.ModuleList([
            ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
                          layer_idx, reduction_factor, opt, use_bn,
                          use_gate, is_multimodal=False)
            for _ in range(self.num_singlemodal_experts)
        ])

        self.router = nn.Sequential(
            nn.Linear(input_dim + linear_out, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_multimodal_experts + self.num_singlemodal_experts)
        )

    def forward(self, x, vis_token=None):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token_fc = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token_fc.permute(0, 2, 1).unsqueeze(-1)
        modal_1 = x.squeeze(-1).permute(0, 2, 1)
        modal_2 = vis_token_fc
        modal_1 = modal_1.mean(dim=1, keepdim=True)
        modal_2 = modal_2.mean(dim=1, keepdim=True)
        multimodal_input = torch.cat((modal_1, modal_2), dim=-1)
        gating_logits = self.router(multimodal_input)
        gating_probs = F.softmax(gating_logits, dim=-1)

        expert_indices = torch.argmax(gating_probs, dim=-1)
        expert_outputs = []
        for expert in self.multimodal_experts + self.singlemodal_experts:
            expert_output = expert(x, vis_token)
            expert_outputs.append(expert_output)
        expert_outputs_tensor = torch.concat(expert_outputs, dim=-1)
        final_expert_output = (expert_outputs_tensor * gating_probs.unsqueeze(-2)).sum(dim=-1, keepdim=True)
        return final_expert_output, expert_indices


def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


class AVQA_Fusion_Net(nn.Module):

    def __init__(self, opt):
        super(AVQA_Fusion_Net, self).__init__()

        self.opt = opt

        # for features
        self.fc_a1 = nn.Linear(768, 1536)
        self.fc_a2 = nn.Linear(1536, 1536)

        self.fc_a1_pure = nn.Linear(768, 1536)
        self.fc_a2_pure = nn.Linear(1536, 1536)

        self.fc_fusion = nn.Linear(1536 + 1536, 1536)

        self.linear11 = nn.Linear(1536, 1536)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(1536, 1536)

        self.linear21 = nn.Linear(1536, 1536)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(1536, 1536)
        self.norm1 = nn.LayerNorm(1536)
        self.norm2 = nn.LayerNorm(1536)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(1536)

        self.attn_a = nn.MultiheadAttention(1536, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(1536, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 1536, 1536, 1, 1536)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(1536, self.opt.avqa_fc_class) # 42

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl = nn.Linear(1536 + 1536, 1536)

        # combine
        self.fc1 = nn.Linear(1536 + 1536, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()

        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)

        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 50
            esc_config.checkpoint_path = "./../checkpoints/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320
            esc_config.classes_num = 527
            esc_config.checkpoint_path = "./../checkpoints/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = "./../checkpoints/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
        else:
            raise NotImplementedError

        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_config.htsat_spec_size,
            patch_size=esc_config.htsat_patch_size,
            in_chans=1,
            num_classes=esc_config.classes_num,
            window_size=esc_config.htsat_window_size,
            config=esc_config,
            depths=esc_config.htsat_depth,
            embed_dim=esc_config.htsat_dim,
            patch_stride=esc_config.htsat_stride,
            num_heads=esc_config.htsat_num_head
        )

        checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        tmp = {k[10:]: v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(tmp, strict=True)

        # self.nce_av = InfoNCELoss(margin=opt.tmp_av)
        # self.nce_tv = InfoNCELoss(margin=opt.tmp_tv)

        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []

        ### ------------> for swin and htsat
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            if self.opt.num_skip>1:
                if (idx_layer+1)%self.opt.num_skip ==0:
                    continue
            conv_dim_tmp = (my_blk.input_resolution[0] * my_blk.input_resolution[1])
            conv_dim_tmp_a = (my_blk_a.input_resolution[0] * my_blk_a.input_resolution[1])
            if not isinstance(my_blk.downsample, nn.Identity):
                down_in_dim.append(my_blk.downsample.reduction.in_features)
                down_out_dim.append(my_blk.downsample.reduction.out_features)
            if my_blk_a.downsample is not None:
                down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
                down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)

            for blk, blk_a in zip(my_blk.blocks, my_blk_a.blocks):
                hidden_d_size = blk.norm1.normalized_shape[0]
                hidden_list.append(hidden_d_size)
                conv_dim.append(conv_dim_tmp)
                hidden_d_size_a = blk_a.norm1.normalized_shape[0]
                hidden_list_a.append(hidden_d_size_a)
                conv_dim_a.append(conv_dim_tmp_a)
        ### <--------------

        self.audio_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i],
                       adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i,
                       reduction_factor=self.opt.Adapter_downsample,
                       opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i],
                       conv_dim_out=conv_dim_a[i],
                       linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list_a))])

        self.vis_adapter_blocks_p1 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                       dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                       linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list))])

        self.audio_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",
                       dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i],
                       conv_dim_out=conv_dim_a[i],
                       linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p2 = nn.ModuleList([
            MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                       dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                       use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                       linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list_a))])

    def forward(self, audio, visual_posi, visual_nega, question, mixup_lambda, stage='eval'):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        bs, t, c, h, w = visual_posi.shape

        audio = audio.view(audio.size(0) * audio.size(1), -1)
        waveform = audio
        bs = visual_posi.size(0)

        visual_posi = rearrange(visual_posi, 'b t c w h -> (b t) c w h')
        f_v = self.swin.patch_embed(visual_posi)
        # f_v_neg = self.swin.patch_embed(visual_nega)

        audio = self.htsat.spectrogram_extractor(audio)
        audio = self.htsat.logmel_extractor(audio)
        audio = audio.transpose(1, 3)
        audio = self.htsat.bn0(audio)
        audio = audio.transpose(1, 3)
        if self.htsat.training:
            audio = self.htsat.spec_augmenter(audio)
        if self.htsat.training and mixup_lambda is not None:
            audio = do_mixup(audio, mixup_lambda)

        if audio.shape[2] > self.htsat.freq_ratio * self.htsat.spec_size:
            audio = self.htsat.crop_wav(audio, crop_size=self.htsat.freq_ratio * self.htsat.spec_size)
            audio = self.htsat.reshape_wav2img(audio)
        else:  # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a)

        idx_layer = 0
        multi_scale = []

        idx_block = 0
        adapter_index_dict = {'audio': {'p1': [], 'p2': []}, 'video': {'p1': [], 'p2': []}}

        for layer_index, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None,
                              htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4],
                              None, None, htsat_blk.blocks[5]]
                assert len(aud_blocks) == len(my_blk.blocks)

            for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:
                    if self.opt.num_skip > 1 and (((layer_index+1) % self.opt.num_skip)==0):
                        f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                        f_a, _ = blk_a(f_a)
                        f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                    else:
                        if self.opt.is_audio_adapter_p1:
                            f_a_res, f_a_moe_adapter_index_p1 = self.audio_adapter_blocks_p1[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p1  = self.vis_adapter_blocks_p1[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1), f_a.permute(0, 2, 1).unsqueeze(-1))

                            adapter_index_dict['audio']['p1'].append(f_a_moe_adapter_index_p1.squeeze().tolist())
                            adapter_index_dict['video']['p1'].append(f_v_moe_adapter_index_p1.squeeze().tolist())
                            f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                            f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                        f_a, _ = blk_a(f_a)
                        if self.opt.is_audio_adapter_p1:
                            f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                        if self.opt.is_audio_adapter_p2:
                            f_a_res, f_a_moe_adapter_index_p2 = self.audio_adapter_blocks_p2[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1), f_v.permute(0, 2, 1).unsqueeze(-1))
                            f_v_res, f_v_moe_adapter_index_p2 = self.vis_adapter_blocks_p2[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1), f_a.permute(0, 2, 1).unsqueeze(-1))
                            adapter_index_dict['audio']['p2'].append(f_a_moe_adapter_index_p2.squeeze().tolist())
                            adapter_index_dict['video']['p2'].append(f_v_moe_adapter_index_p2.squeeze().tolist())

                        f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                        if self.opt.is_audio_adapter_p2:
                            f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)
                            f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                        idx_layer = idx_layer + 1
                else:
                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))

            #####
            f_v = my_blk.downsample(f_v)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)

        f_v = self.swin.norm(f_v)

        with torch.no_grad():

            visual_nega = rearrange(visual_nega, 'b t c h w -> (b t) c h w')
            visual_nega = self.swin.forward_features(visual_nega)

        ############## <----------

        visual_posi = rearrange(f_v, '(b t) (h w) c -> b t c h w', b=bs, t=t, h=6, w=6)
        visual_nega = rearrange(visual_nega, '(b t) (h w) c -> b t c h w', b=bs, t=t, h=6, w=6)

        f_a = f_a.mean(dim=1)
        # f_a = torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1)
        audio = rearrange(f_a, '(b t) c -> b t c', b=bs, t=t)
        ### <-----

        # visual_posi, f_a = self.temporal_attn(visual_posi, f_a)

        ## question features
        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ## audio features  [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat_pure = audio_feat
        B, T, C = audio_feat.size()  # [B, T, C]
        audio_feat = audio_feat.view(B * T, C)  # [B*T, C]

        ## visual posi [2*B*T, C, H, W]
        B, T, C, H, W = visual_posi.size()
        temp_visual = visual_posi.view(B * T, C, H, W)  # [B*T, C, H, W]
        v_feat = self.avgpool(temp_visual)  # [B*T, C, 1, 1]
        visual_feat_before_grounding_posi = v_feat.squeeze()  # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)  # [B, HxW, C]

        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)  # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)  # [B*T, C, 1]

        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze()  # [B*T, HxW]

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_posi)
        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()  # [B*T, C]

        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)  # [B*T, C]

        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)  # [B*T, C*2], [B*T, 3072]

        feat = F.relu(self.fc1(feat))  # (3072, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_posi = self.fc4(feat)  # (128, 2)

        ###############################################################################################
        # visual nega
        B, T, C, H, W = visual_nega.size()
        temp_visual = visual_nega.view(B * T, C, H, W)
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding_nega = v_feat.squeeze()  # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)  # [B, HxW, C]
        visual_feat_nega = nn.functional.normalize(v_feat, dim=2)

        ##### av grounding nega
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)  # [B*T, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat_nega)
        visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()  # [B*T, C]

        visual_gl = torch.cat((visual_feat_before_grounding_nega, visual_feat_grd_after_grounding_nega), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_nega = self.fc_gl(visual_feat_grd)  # [B*T, C]

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)  # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))  # (1024, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match_nega = self.fc4(feat)  # (128, 2)

        ###############################################################################################

        # out_match=None
        # match_label=None

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 1536)  # [B, T, 512]
        visual_feat_grd = visual_feat_grd_be.permute(1, 0, 2)

        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[
            0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)

        # attention, question as query on audio
        audio_feat_be = audio_feat_pure.view(B, -1, 1536)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)

        feat = torch.cat((audio_feat_att + audio_feat_be.mean(dim=-2).squeeze(),
                          visual_feat_att + visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size]

        return out_qa, out_match_posi, out_match_nega, adapter_index_dict

