import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from model.utils import do_mixup, get_mix_lambda, do_mixup_label

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb
from base_options import BaseOptions


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

if __name__ == "__main__":
    args = BaseOptions()
    args.parser.add_argument("--test_batch_size", default=1, type=int)
    args.parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    args = args.parse()
    args.log_dir = './test_logs'
    
    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel_v2 as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")


    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        opt=args, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    split = 'test'
    test_dataset = S4Dataset(split, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    num_experts_per_layer = args.num_multimodal_experts + args.num_singlemodal_experts
    num_layers = 12 
    audio_p1_expert_activation_counts = np.zeros((num_layers, num_experts_per_layer))
    audio_p2_expert_activation_counts = np.zeros((num_layers, num_experts_per_layer))
    video_p1_expert_activation_counts = np.zeros((num_layers, num_experts_per_layer))
    video_p2_expert_activation_counts = np.zeros((num_layers, num_experts_per_layer))

    audio_p1_expert_activation_probs = np.zeros((num_layers, num_experts_per_layer))
    audio_p2_expert_activation_probs = np.zeros((num_layers, num_experts_per_layer))
    video_p1_expert_activation_probs = np.zeros((num_layers, num_experts_per_layer))
    video_p2_expert_activation_probs = np.zeros((num_layers, num_experts_per_layer))
    visual_features_list = []
    audio_features_list = []
    audio_features_v2_list = []

    labels_list = []

    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio_spec, audio, mask, wave, category_list, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            if args.visual_only==1:
                wave = torch.zeros_like(wave, requires_grad=False)
            if args.backbone_type == "audioset":
                mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*5)).to('cuda')
            else:
                mixup_lambda = None
            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            # imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)

            output, _, _, adapter_index_dict, adapter_probs, _, visual_feature, audio_feature, audio_feature_v2 = model(imgs, wave, mixup_lambda=mixup_lambda, is_training=False) # [5, 1, 224, 224] = [bs=1 * T=5, 1, 224, 224]
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list)

            visual_features_list.append(visual_feature.cpu().numpy())
            audio_features_list.append(audio_feature.cpu().numpy())
            audio_features_v2_list.append(audio_feature_v2.cpu().numpy())

            labels_list.append(category_list*5)


            audio_adapter_p1_index_list = adapter_index_dict["audio"]["p1"]
            audio_adapter_p2_index_list = adapter_index_dict["audio"]["p2"]
            video_adapter_p1_index_list = adapter_index_dict["video"]["p1"]
            video_adapter_p2_index_list = adapter_index_dict["video"]["p2"]

            audio_adapter_p1_probs_list = adapter_probs["audio"]["p1"]
            audio_adapter_p2_probs_list = adapter_probs["audio"]["p2"]
            video_adapter_p1_probs_list = adapter_probs["video"]["p1"]
            video_adapter_p2_probs_list = adapter_probs["video"]["p2"]

            real_num_layers = len(audio_adapter_p1_index_list)
            for layer in range(real_num_layers):
                audio_adapter_p1_unique, audio_adapter_p1_counts = np.unique(audio_adapter_p1_index_list[layer], return_counts=True)
                audio_adapter_p2_unique, audio_adapter_p2_counts = np.unique(audio_adapter_p2_index_list[layer], return_counts=True)
                video_adapter_p1_unique, video_adapter_p1_counts = np.unique(video_adapter_p1_index_list[layer], return_counts=True)
                video_adapter_p2_unique, video_adapter_p2_counts = np.unique(video_adapter_p2_index_list[layer], return_counts=True)

                audio_p1_expert_activation_counts[layer, audio_adapter_p1_unique] += audio_adapter_p1_counts
                audio_p2_expert_activation_counts[layer, audio_adapter_p2_unique] += audio_adapter_p2_counts
                video_p1_expert_activation_counts[layer, video_adapter_p1_unique] += video_adapter_p1_counts
                video_p2_expert_activation_counts[layer, video_adapter_p2_unique] += video_adapter_p2_counts

                audio_p1_expert_activation_probs[layer] += np.array(audio_adapter_p1_probs_list[layer]).sum(axis=0)
                audio_p2_expert_activation_probs[layer] += np.array(audio_adapter_p2_probs_list[layer]).sum(axis=0)
                video_p1_expert_activation_probs[layer] += np.array(video_adapter_p1_probs_list[layer]).sum(axis=0)
                video_p2_expert_activation_probs[layer] += np.array(video_adapter_p2_probs_list[layer]).sum(axis=0)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))

        sample_size = len(audio_adapter_p1_index_list[0])
        num_iterations = len(test_dataloader)
        total_iterations = num_iterations * sample_size

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))

        logger.info("expert_activation_counts")
        expert_save_dir = args.log_dir
        logger.info("expert_save_dir :{}".format(expert_save_dir))
        np.save(os.path.join(expert_save_dir,'audio_p1_expert_activation_counts.npy'), audio_p1_expert_activation_counts)
        np.save(os.path.join(expert_save_dir,'audio_p2_expert_activation_counts.npy'), audio_p2_expert_activation_counts)
        np.save(os.path.join(expert_save_dir,'video_p1_expert_activation_counts.npy'), video_p1_expert_activation_counts)
        np.save(os.path.join(expert_save_dir,'video_p2_expert_activation_counts.npy'), video_p2_expert_activation_counts)

        np.save(os.path.join(expert_save_dir,'audio_p1_expert_activation_probs.npy'), audio_p1_expert_activation_probs)
        np.save(os.path.join(expert_save_dir,'audio_p2_expert_activation_probs.npy'), audio_p2_expert_activation_probs)
        np.save(os.path.join(expert_save_dir,'video_p1_expert_activation_probs.npy'), video_p1_expert_activation_probs)
        np.save(os.path.join(expert_save_dir,'video_p2_expert_activation_probs.npy'), video_p2_expert_activation_probs)

        np.save(os.path.join(expert_save_dir,'total_iterations.npy'), total_iterations)
        visual_features = np.vstack(visual_features_list)
        audio_features = np.vstack(audio_features_list)
        audio_features_v2 = np.vstack(audio_features_v2_list)

        labels_features = labels_list # np.vstack(labels_list)
        
        os.makedirs(os.path.join(expert_save_dir, 'tsne'), exist_ok=True)
        np.save(os.path.join(expert_save_dir, 'tsne' ,'visual_features.npy'), visual_features)
        np.save(os.path.join(expert_save_dir,'tsne', 'audio_features.npy'), audio_features)
        np.save(os.path.join(expert_save_dir,'tsne', 'audio_features_v2.npy'), audio_features_v2)
        np.save(os.path.join(expert_save_dir,'tsne', 'label_features.npy'), labels_features)











