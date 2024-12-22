import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
from model.utils import do_mixup, get_mix_lambda, do_mixup_label

# from gpuinfo import GPUInfo 
from base_options import BaseOptions
args = BaseOptions().parse()

# mygpu = GPUInfo.get_info()[0]
# gpu_source = {}

# if 'N/A' in mygpu.keys():
# 	for info in mygpu['N/A']:
# 		if info in gpu_source.keys():
# 			gpu_source[info] +=1
# 		else:
# 			gpu_source[info] =1

# for gpu_id in args.gpu:
# 	gpu_id = str(gpu_id)

# 	if gpu_id not in gpu_source.keys():
# 		print('go gpu:', gpu_id)
# 		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# 		break
# 	elif gpu_source[gpu_id] < 1:
# 		print('go gpu:', gpu_id)
# 		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# 		break
# os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
import logging

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou,  Eval_Fmeasure, save_mask
from utils.system import setup_logging
from ipdb import set_trace

import certifi
import sys

import wandb

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())



class audio_extractor(torch.nn.Module):
	def __init__(self, cfg, device):
		super(audio_extractor, self).__init__()
		self.audio_backbone = vggish.VGGish(cfg, device)

	def forward(self, audio):
		audio_fea = self.audio_backbone(audio)
		return audio_fea


if __name__ == "__main__":
	# parser = argparse.ArgumentParser()    

	# args = parser.parse_args()

	
	if args.wandb:

		wandb.init(config=args, project="ada_av_segmentation", name=args.model_name)

	if (args.visual_backbone).lower() == "resnet":
		from model import ResNet_AVSModel as AVSModel
		print('==> Use ResNet50 as the visual backbone...')
	elif (args.visual_backbone).lower() == "pvt":
		from model import PVT_AVSModel_v2 as AVSModel
		print('==> Use pvt-v2 as the visual backbone...')
	else:
		raise NotImplementedError("only support the resnet50 and pvt-v2")


	# Fix seed
	FixSeed = 123
	random.seed(FixSeed)
	np.random.seed(FixSeed)
	torch.manual_seed(FixSeed)
	torch.cuda.manual_seed(FixSeed)

	# Log directory
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir, exist_ok=True)
	# Logs
	prefix = args.session_name
	log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
	args.log_dir = log_dir

	# Save scripts
	script_path = os.path.join(log_dir, 'scripts')
	if not os.path.exists(script_path):
		os.makedirs(script_path, exist_ok=True)

	scripts_to_save = ['train.sh', 'train_v2.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel_v2.py', 'loss.py']
	for script in scripts_to_save:
		dst_path = os.path.join(script_path, script)
		try:
			shutil.copy(script, dst_path)
		except IOError:
			os.makedirs(os.path.dirname(dst_path), exist_ok=True)
			shutil.copy(script, dst_path)

	# Checkpoints directory
	checkpoint_dir = os.path.join(log_dir, 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir, exist_ok=True)
	args.checkpoint_dir = checkpoint_dir

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
	model = torch.nn.DataParallel(model).cuda()
	model.train()

	total_params = 0
	train_params = 0
	additional_params = 0
	for name, param in model.named_parameters():
	
		param.requires_grad = True
		### ---> compute params
		tmp = 1
		for num in param.shape:
			tmp *= num
		# if 'encoder_backbone' not in name:
		total_params += tmp
	
		if 'ViT'in name or 'swin' in name:
			param.requires_grad = False
		elif 'adapter' in name:
			additional_params += tmp
			train_params += tmp
		elif 'temporal_attn' in name:
			additional_params += tmp
			train_params += tmp
		else:
			train_params += tmp

	
	print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
	print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
	print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


	# for k, v in model.named_parameters():
	#         print(k, v.requires_grad)

	# video backbone
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	audio_backbone = audio_extractor(cfg, device)

	# audio_backbone = torch.nn.DataParallel(audio_backbone).cuda() 
	audio_backbone.cuda()
	audio_backbone.eval()

	total_params_audio = 0
	for name, param in audio_backbone.named_parameters():
	
		tmp = 1
		for num in param.shape:
			tmp *= num

		total_params_audio += tmp
	

	
	print('####### Total audio params in M: %0.1f M  #######'%(total_params_audio/1000000))

	# Data
	train_dataset = S4Dataset('train', args)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
														batch_size=args.train_batch_size,
														shuffle=True,
														num_workers=args.num_workers,
														pin_memory=True)
	max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

	val_dataset = S4Dataset('test',args)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,
														batch_size=args.val_batch_size,
														shuffle=False,
														num_workers=args.num_workers,
														pin_memory=True)

	# Optimizer
	model_params = model.parameters()
	optimizer = torch.optim.Adam(model_params, args.lr)


	avg_meter_total_loss = pyutils.AverageMeter('total_loss')
	avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
	avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
	avg_meter_miou = pyutils.AverageMeter('miou')
	avg_meter_F = pyutils.AverageMeter('F_score')

	# Train
	best_epoch = 0
	global_step = 0
	miou_list = []
	fscore_list = []
	max_miou = 0
	for epoch in range(args.max_epoches):
		for n_iter, batch_data in enumerate(train_dataloader):
			# imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
			imgs, audio_spec, audio, mask, wave = batch_data
			if args.backbone_type == "audioset":
				mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*5)).to('cuda')
			else:
				mixup_lambda = None
			imgs = imgs.cuda()
			audio = audio.cuda()
			mask = mask.cuda()
			B, frame, C, H, W = imgs.shape
			# imgs = imgs.view(B*frame, C, H, W)
			mask = mask.view(B, H, W)

			audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]

			output, visual_map_list, a_fea_list, adapter_index_dict, adapter_probs, load_balancing_loss = model(imgs, wave, mixup_lambda=mixup_lambda, is_training=True)
			loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1), \
												a_fea_list, visual_map_list, \
												lambda_1=args.lambda_1, \
												count_stages=args.sa_loss_stages, \
												sa_loss_flag=args.sa_loss_flag, \
												mask_pooling_type=args.mask_pooling_type)

			if args.use_load_balacing_loss == 1:
				loss += load_balancing_loss * args.load_balancing_loss_weight
				
			avg_meter_total_loss.add({'total_loss': loss.item()})
			avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
			avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

			loss.backward()

			if ((n_iter + 1) % args.accum_itr == 0) or (n_iter + 1 == len(train_dataloader)):
				optimizer.step()
				optimizer.zero_grad() 

			global_step += 1

			if (global_step-1) % 50 == 0:
				train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, load_balance_loss:%.4f, lambda_1:%.4f, lr: %.6f'%(
							global_step-1, 
							max_step, 
							avg_meter_total_loss.pop('total_loss'), 
							avg_meter_iou_loss.pop('iou_loss'), 
							avg_meter_sa_loss.pop('sa_loss'), 
							load_balancing_loss,
							args.lambda_1, 
							optimizer.param_groups[0]['lr'])

				logger.info(train_log)


		# Validation:
		count = 0 
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

		model.eval()
		with torch.no_grad():
			for n_iter, batch_data in enumerate(val_dataloader):
				imgs, audio_spec, audio, mask, wave, _, _ = batch_data
				if args.backbone_type == "audioset":
					mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
				else:
					mixup_lambda = None  

				imgs = imgs.cuda()
				audio = audio.cuda()
				mask = mask.cuda()
				B, frame, C, H, W = imgs.shape
				# imgs = imgs.view(B*frame, C, H, W)
				mask = mask.view(B*frame, H, W)

				output, _, _ , adapter_index_dict, adapter_probs, _ = model(imgs, wave, mixup_lambda=mixup_lambda, is_training=False) # [bs*5, 1, 224, 224]

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

			sample_size = len(audio_adapter_p1_index_list[0])
			num_iterations = len(val_dataloader)
			total_iterations = num_iterations * sample_size

			miou = (avg_meter_miou.pop('miou'))
			F_score = (avg_meter_F.pop('F_score'))
			print('val miou:', miou.item())
			print('val F_score:', F_score)

			count = count +1
			if miou > max_miou:
				model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
				torch.save(model.module.state_dict(), model_save_path)
				best_epoch = epoch
				logger.info('save best model to %s'%model_save_path)
				count = 0

				if args.wandb:
					wandb.log({"val-best": miou})
			if count == args.early_stop:
				exit()

			miou_list.append(miou)
			fscore_list.append(F_score)

			max_miou = max(miou_list)
			max_fscore = max(fscore_list)

			val_log = 'Epoch: {}, Miou: {}, F_score: {}, maxMiou: {}, maxF_Score: {}'.format(epoch, miou, F_score, max_miou, max_fscore)
			# print(val_log)
			logger.info(val_log)
			logger.info("expert_activation_counts")
			expert_save_dir = args.log_dir
			np.save(os.path.join(expert_save_dir,'audio_p1_expert_activation_counts.npy'), audio_p1_expert_activation_counts)
			np.save(os.path.join(expert_save_dir,'audio_p2_expert_activation_counts.npy'), audio_p2_expert_activation_counts)
			np.save(os.path.join(expert_save_dir,'video_p1_expert_activation_counts.npy'), video_p1_expert_activation_counts)
			np.save(os.path.join(expert_save_dir,'video_p2_expert_activation_counts.npy'), video_p2_expert_activation_counts)

			np.save(os.path.join(expert_save_dir,'audio_p1_expert_activation_probs.npy'), audio_p1_expert_activation_probs)
			np.save(os.path.join(expert_save_dir,'audio_p2_expert_activation_probs.npy'), audio_p2_expert_activation_probs)
			np.save(os.path.join(expert_save_dir,'video_p1_expert_activation_probs.npy'), video_p1_expert_activation_probs)
			np.save(os.path.join(expert_save_dir,'video_p2_expert_activation_probs.npy'), video_p2_expert_activation_probs)

			np.save(os.path.join(expert_save_dir,'total_iterations.npy'), total_iterations)

		model.train()
	logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
	logger.info('best val F_score {} at peoch: {}'.format(max_fscore, best_epoch))











