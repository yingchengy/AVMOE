import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nets.grouping import ModalityTrans
from torch.autograd import Variable
import numpy as np
import copy
import math

from ipdb import set_trace
import os

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention
import timm
import loralib as lora
from transformers.activations import get_activation
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint
from .models import EncoderLayer, DecoderLayer, Decoder
from .models import Encoder as CMBS_Encoder

from .htsat import HTSAT_Swin_Transformer
import nets.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ExpertAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, opt=None, is_multimodal=True):
		super().__init__()
		self.opt = opt
		self.adapter_kind = adapter_kind
		self.reduction_factor = self.opt.Adapter_downsample
		self.use_bn = self.opt.is_bn
		self.use_gate=self.opt.is_gate
		self.is_multimodal = is_multimodal
		self.num_tk = self.opt.num_tokens

		if self.use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		# bottleneck, multi_modal_adapter
		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // self.reduction_factor
			self.my_tokens = nn.Parameter(torch.rand((self.num_tk, input_dim)))

			self.gate_av = nn.Parameter(torch.zeros(1))
			self.activation = nn.ReLU(inplace=True)
			
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, 
										groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, 
							   			groups=self.opt.num_conv_group, bias=False)

			if self.use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
		
		# bottleneck, single_modal_adapter
		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // self.reduction_factor
			self.gate_av = nn.Parameter(torch.zeros(1))
			self.activation = nn.ReLU(inplace=True)

			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, 
										groups=self.opt.num_conv_group, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, 
							   			groups=self.opt.num_conv_group, bias=False)

			if self.use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
		
		# error
		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))			
			att_v2tk = F.softmax(att_v2tk, dim=-1)			
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))			
			rep_token = rep_token + rep_token_res

			# cross-modal attention
			att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))
			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

			x = x + self.gate_av * x_res.contiguous()
			
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
			# self attention
			x_squ = x.squeeze(-1)
			att_tk2x = torch.bmm(x_squ.permute(0, 2, 1), x_squ)
			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(x_squ, att_tk2x).unsqueeze(-1)
			
			x = x + self.gate_av * x_res.contiguous()

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
		
		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)
		
		if self.gate is not None:
			output = self.gate * output
		
		return output


class MoEAdapter(nn.Module):
	def __init__(self, input_dim, output_dim, adapter_kind, dim_list, layer_idx, opt=None, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
		super().__init__()
		self.opt = opt
		self.num_multimodal_experts = self.opt.num_multimodal_experts
		self.num_singlemodal_experts = self.opt.num_singlemodal_experts
		
		self.multimodal_experts = nn.ModuleList([
			ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
						  layer_idx, opt, is_multimodal=True)
			for _ in range(self.num_multimodal_experts)
		])
		
		self.singlemodal_experts = nn.ModuleList([
			ExpertAdapter(input_dim, output_dim, adapter_kind, dim_list,
						  layer_idx, opt, is_multimodal=False)
			for _ in range(self.num_singlemodal_experts)
		])

		self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)
		self.fc = nn.Linear(linear_in, linear_out)
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

		expert_outputs = []
		for expert in self.multimodal_experts:
			expert_output = expert(x, vis_token)
			expert_outputs.append(expert_output)
		for expert in self.singlemodal_experts:
			expert_output = expert(x, vis_token)
			expert_outputs.append(expert_output)
		expert_outputs_tensor = torch.concat(expert_outputs, dim=-1)

		multimodal_input = torch.cat((modal_1, modal_2), dim=-1)
		gating_logits = self.router(multimodal_input)
		gating_probs = F.softmax(gating_logits, dim=-1)
		final_expert_output = (expert_outputs_tensor * gating_probs.unsqueeze(-2)).sum(dim=-1, keepdim=True)

		if self.opt.use_load_balacing_loss==1:
			load_balancing_loss = self.compute_load_balancing_loss(gating_probs)
		else:
			load_balancing_loss = 0.

		return final_expert_output, load_balancing_loss

	def compute_load_balancing_loss(self, gating_probs):
		expert_probs_mean = torch.mean(gating_probs, dim=0)
		uniform_distribution = torch.full_like(expert_probs_mean, 1.0 / expert_probs_mean.size(0))
		load_balancing_loss = F.kl_div(expert_probs_mean.log(), uniform_distribution, reduction='batchmean')
		return load_balancing_loss


class MGN_Net(nn.Module):

    def __init__(self, args):
        super(MGN_Net, self).__init__()

        opt = args
        self.opt = opt
        self.fc_a =  nn.Linear(768, args.dim)
        self.fc_v = nn.Linear(1536, args.dim)
        self.fc_st = nn.Linear(512, args.dim)
        self.fc_fusion = nn.Linear(args.dim * 2, args.dim)

        # hard or soft assignment
        self.unimodal_assgin = args.unimodal_assign
        self.crossmodal_assgin = args.crossmodal_assign

        unimodal_hard_assignment = True if args.unimodal_assign == 'hard' else False
        crossmodal_hard_assignment = True if args.crossmodal_assign == 'hard' else False

        # learnable tokens
        self.audio_token = nn.Parameter(torch.zeros(25, args.dim))
        self.visual_token = nn.Parameter(torch.zeros(25, args.dim))

        # class-aware uni-modal grouping
        self.audio_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_aud,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=True
                        )

        self.visual_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_vis,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=False
                        )

        # modality cross-modal grouping
        self.av_mcg = ModalityTrans(
                            args.dim,
                            depth=args.depth_av,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=crossmodal_hard_assignment,
                            use_han=False                        
                        )

        # prediction
        self.fc_prob = nn.Linear(args.dim, 1)
        self.fc_prob_a = nn.Linear(args.dim, 1)
        self.fc_prob_v = nn.Linear(args.dim, 1)

        self.fc_cls = nn.Linear(args.dim, 25)

        self.apply(self._init_weights)

        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
        self.checkpoint_path = opt.checkpoint_path
        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 50
            esc_config.checkpoint_path =  self.checkpoint_path + "/ESC-50/"
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 527
            esc_config.checkpoint_path = self.checkpoint_path + "/AudioSet/"
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = self.checkpoint_path + "/SCV2/"
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
        else:
            raise NotImplementedError
    
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_config.htsat_spec_size,
            patch_size=esc_config.htsat_patch_size,
            in_chans=1,
            num_classes=esc_config.classes_num,
            window_size=esc_config.htsat_window_size,
            config = esc_config,
            depths = esc_config.htsat_depth,
            embed_dim = esc_config.htsat_dim,
            patch_stride=esc_config.htsat_stride,
            num_heads=esc_config.htsat_num_head
        )
        
        checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        tmp = {k[10:]:v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(tmp, strict=True)
        
        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []
        
        ## ------------> for swin and htsat 
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
            conv_dim_tmp = (my_blk.input_resolution[0]*my_blk.input_resolution[1])
            conv_dim_tmp_a = (my_blk_a.input_resolution[0]*my_blk_a.input_resolution[1])
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


        if self.opt.is_audio_adapter_p1:
            self.audio_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                            linear_in=hidden_list[i], linear_out=hidden_list_a[i]       
                            )
                for i in range(len(hidden_list_a))])

            self.vis_adapter_blocks_p1 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                            linear_in=hidden_list_a[i], linear_out=hidden_list[i]       
                            )
                for i in range(len(hidden_list))])

        if self.opt.is_audio_adapter_p2:
            self.audio_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
                            linear_in=hidden_list[i], linear_out=hidden_list_a[i]       
                            )
                for i in range(len(hidden_list_a))])

            self.vis_adapter_blocks_p2 = nn.ModuleList([
                MoEAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], 
                            adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i, opt=opt,  
							conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
                            linear_in=hidden_list_a[i], linear_out=hidden_list[i]       
                            )
                for i in range(len(hidden_list))])
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, audio, visual, visual_st, mixup_lambda=None):
        b, t, d = visual_st.size()
        
        audio = audio.view(audio.size(0)*audio.size(1), -1)
        waveform = audio
        bs = visual.size(0)
        vis = rearrange(visual, 'b t c w h -> (b t) c w h')
        f_v = self.swin.patch_embed(vis)
        
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
        else: # this part is typically used, and most easy one
            audio = self.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.htsat.patch_embed(audio)
        if self.htsat.ape:
            f_a = f_a + self.htsat.absolute_pos_embed
        f_a = self.htsat.pos_drop(f_a)
        
        idx_layer = 0
        out_idx_layer = 0
        total_load_balancing_loss = 0
        for _, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)) :

            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None, htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4], None, None, htsat_blk.blocks[5]]
                assert len(aud_blocks) == len(my_blk.blocks)
                
            for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:
                        
                    f_a_res, load_balancing_loss_a = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
                    f_v_res, load_balancing_loss_v = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))
                    total_load_balancing_loss += load_balancing_loss_a + load_balancing_loss_v

                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

                    f_a, _ = blk_a(f_a)
                    f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
            
                    f_a_res, load_balancing_loss_a = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
                    f_v_res, load_balancing_loss_v = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))
                    total_load_balancing_loss += load_balancing_loss_a + load_balancing_loss_v

                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

                    f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
                    
                    idx_layer = idx_layer +1
                    
                else:
                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))

            f_v = my_blk.downsample(f_v)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)


        f_v = self.swin.norm(f_v)
        
        # ly: Processing similar to spatial attention mechanisms
        f_v = f_v.mean(dim=1, keepdim=True).permute(1, 0, 2) # [B, 10, 1536]
        f_a = f_a.mean(dim=1, keepdim=True).permute(1, 0, 2) # [B, 10, 768]
        
        x1_0 = self.fc_a(f_a)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(f_v)
        vid_st = self.fc_st(visual_st)

        x2_0 = torch.cat((vid_s, vid_st), dim=-1)
        x2_0 = self.fc_fusion(x2_0)

        # visual uni-modal grouping
        x2, attn_visual_dict, _ = self.visual_cug(x2_0, self.visual_token, return_attn=True)

        # audio uni-modal grouping
        x1, attn_audio_dict, _ = self.audio_cug(x1_0, self.audio_token, x2_0, return_attn=True)

        # modality-aware cross-modal grouping
        x, _, _ = self.av_mcg(x1, x2, return_attn=True)

        
        # prediction
        av_prob = torch.sigmoid(self.fc_prob(x))                                # [B, 25, 1]
        global_prob = av_prob.sum(dim=-1)                                       # [B, 25]

        # cls token prediction
        aud_cls_prob = self.fc_cls(self.audio_token)                            # [25, 25]
        vis_cls_prob = self.fc_cls(self.visual_token)                           # [25, 25]

        # attentions
        attn_audio = attn_audio_dict[self.unimodal_assgin].squeeze(1)                    # [25, 10]
        attn_visual = attn_visual_dict[self.unimodal_assgin].squeeze(1)                  # [25, 10]

        # audio prediction
        a_prob = torch.sigmoid(self.fc_prob_a(x1))                                # [B, 25, 1]
        a_frame_prob = (a_prob * attn_audio).permute(0, 2, 1)                     # [B, 10, 25]
        a_prob = a_prob.sum(dim=-1)                                               # [B, 25]

        # visual prediction
        v_prob = torch.sigmoid(self.fc_prob_v(x2))                                # [B, 25, 1]
        v_frame_prob = (v_prob * attn_visual).permute(0, 2, 1)                    # [B, 10, 25]
        v_prob = v_prob.sum(dim=-1)                                               # [B, 25]

        return aud_cls_prob, vis_cls_prob, global_prob, a_prob, v_prob, a_frame_prob, v_frame_prob, total_load_balancing_loss

