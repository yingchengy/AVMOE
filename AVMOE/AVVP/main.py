from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import pandas as pd
import os
import os.path as osp
import copy
import time
import random
import numpy as np
import wandb

from dataloader import *
from nets.mgn import MGN_Net
from utils.eval_metrics import segment_level, event_level
from nets.utils import do_mixup, get_mix_lambda, do_mixup_label
from einops import rearrange, repeat

current_dir = os.getcwd()
print("current_dir", current_dir)

data_grandparent_dir = '/remote-home/yangli/DG-SCT'
print("data_root_path", data_grandparent_dir)


def train(args, model, train_loader, optimizer, criterion, criterion_ce, epoch):
    model.train()

    for batch_idx, sample in enumerate(train_loader):
        # global_step = batch_idx + len(train_loader) * epoch
        # data
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')

        if args.backbone_type == "audioset":
            mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, 10)).to('cuda')
            mixup_lambda = mixup_lambda.repeat(len(audio), 1)
        else:
            mixup_lambda = None

        aud_cls_prob, vis_cls_prob, output, a_prob, v_prob, _, _, load_balancing_loss = model(audio, video, video_st, mixup_lambda=mixup_lambda)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        # label smoothing
        a = 1.0
        v = 0.9 
        Pa = a * target + (1 - a) * 0.5
        Pv = v * target + (1 - v) * 0.5

        # ly: Modifying loss calculation for multiple GPUs
        num_gpu = len(args.gpu.split(','))
        tmp_tensor = torch.arange(0,25).long().cuda()
        tmp_tensor_copy = tmp_tensor.clone()
        for i in range(num_gpu):
            if i != 0:
                tmp_tensor = torch.cat([tmp_tensor, tmp_tensor_copy], dim=0)
        cls_target = tmp_tensor

        # individual guided learning
        loss_av = criterion(output, target)
        loss_a = criterion(a_prob, Pa)
        loss_v = criterion(v_prob, Pv)

        # class aware prediction
        loss_cls_aud = criterion_ce(aud_cls_prob, cls_target)
        loss_cls_vis = criterion_ce(vis_cls_prob, cls_target)

        loss =  loss_cls_aud + loss_cls_vis + loss_av + loss_a + loss_v

        if args.use_load_balacing_loss==1:
            for lb_loss in load_balancing_loss:
                loss += lb_loss

        loss.backward()

        # weights update
        if ((batch_idx + 1) % args.accum_itr == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # train info
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\nLoss: {:.6f}\tLoss av: {:.6f}\tLoss a: {:.6f}\tLoss v: {:.6f}\tLoss cls_aud: {:.6f}\tLoss cls_vis: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_av.item(), loss_a.item(), loss_v.item(), loss_cls_aud.item(), loss_cls_vis.item()))

            if args.wandb:
                wandb.log({"loss_total": float(loss.item())})
            


def eval(args, model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv(os.path.join(data_grandparent_dir, "data/AVVP/AVVP_eval_audio.csv"), header=0, sep='\t')
    df_v = pd.read_csv(os.path.join(data_grandparent_dir, "data/AVVP/AVVP_eval_visual.csv"), header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')

            if args.backbone_type == "audioset":
                mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(audio)*10)).to('cuda')
            else:
                mixup_lambda = None
                
            _, _, output_list, _, _, a_frame_prob_list, v_frame_prob_list, _  = model(audio, video, video_st, mixup_lambda=mixup_lambda)
            

            # ly: multiple GPUs
            num_gpu = len(args.gpu.split(','))
            for i in range(num_gpu):
                # Splitting the first dimension for multiple GPUs
                if num_gpu == 1:
                    output = output_list
                    a_frame_prob = a_frame_prob_list
                    v_frame_prob = v_frame_prob_list
                else:    
                    output = output_list[i].unsqueeze(0)
                    a_frame_prob = a_frame_prob_list[i].unsqueeze(0)
                    v_frame_prob = v_frame_prob_list[i].unsqueeze(0)

                o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)

                Pa = a_frame_prob[0, :, :].cpu().detach().numpy()
                Pv = v_frame_prob[0, :, :].cpu().detach().numpy()

                # filter out false positive events with predicted weak labels
                Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)         # [10, 25]
                Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)         # [10, 25]

                # extract audio GT labels
                GT_a = np.zeros((25, 10))

                df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
                filenames = df_vid_a["filename"]
                events = df_vid_a["event_labels"]
                onsets = df_vid_a["onset"]
                offsets = df_vid_a["offset"]
                num = len(filenames)
                if num > 0:
                    for i in range(num):
                        x1 = int(onsets[df_vid_a.index[i]])
                        x2 = int(offsets[df_vid_a.index[i]])
                        event = events[df_vid_a.index[i]]
                        idx = id_to_idx[event]
                        GT_a[idx, x1:x2] = 1

                # extract visual GT labels
                GT_v =np.zeros((25, 10))

                df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
                filenames = df_vid_v["filename"]
                events = df_vid_v["event_labels"]
                onsets = df_vid_v["onset"]
                offsets = df_vid_v["offset"]
                num = len(filenames)
                if num > 0:
                    for i in range(num):
                        x1 = int(onsets[df_vid_v.index[i]])
                        x2 = int(offsets[df_vid_v.index[i]])
                        event = events[df_vid_v.index[i]]
                        idx = id_to_idx[event]
                        GT_v[idx, x1:x2] = 1

                GT_av = GT_a * GT_v                 # [25, 10]

                # obtain prediction matrices
                SO_a = np.transpose(Pa)             # [25, 10]
                SO_v = np.transpose(Pv)             # [25, 10]
                SO_av = SO_a * SO_v                 # [25, 10]

                # segment-level F1 scores
                f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
                F_seg_a.append(f_a)
                F_seg_v.append(f_v)
                F_seg.append(f)
                F_seg_av.append(f_av)

                # event-level F1 scores
                f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
                F_event_a.append(f_a)
                F_event_v.append(f_v)
                F_event.append(f)
                F_event_av.append(f_av)
    
    # Segment-level
    p_seg_a = 100 * np.mean(np.array(F_seg_a))
    p_seg_v = 100 * np.mean(np.array(F_seg_v))
    p_seg_av = 100 * np.mean(np.array(F_seg_av))

    print('Audio Event Detection Segment-level F1: {:.1f}'.format(p_seg_a))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(p_seg_v))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(p_seg_av))

    avg_type = (p_seg_a + p_seg_v + p_seg_av)/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-level Type@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    # Event-level
    p_event_a = 100 * np.mean(np.array(F_event_a))
    p_event_v = 100 * np.mean(np.array(F_event_v))
    p_event_av = 100 * np.mean(np.array(F_event_av))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(p_event_a))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(p_event_v))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(p_event_av))

    avg_type_event = (p_event_a + p_event_v + p_event_av) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))

    if args.wandb:
        wandb.log({"p_seg_a acc": p_seg_a})
        wandb.log({"p_seg_v acc": p_seg_v})
        wandb.log({"p_seg_av acc": p_seg_av})
        wandb.log({"avg_type acc": avg_type})
        wandb.log({"avg_event acc": avg_event})

        wandb.log({"p_event_a acc": p_event_a})
        wandb.log({"p_event_v acc": p_event_v})
        wandb.log({"p_event_av acc": p_event_av})
        wandb.log({"avg_type_event acc": avg_type_event})
        wandb.log({"avg_event_level acc": avg_event_level})

    return avg_type, avg_type_event

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Multimodal Grouping Network for AVVP')
    if True:
        parser.add_argument(
            "--audio_dir", type=str, default='data/AVVP/LLP_dataset/feats/vggish/', help="audio dir")
        parser.add_argument(
            "--video_dir", type=str, default='data/AVVP/LLP_dataset/feats/res152/', help="video dir")
        parser.add_argument(
            "--st_dir", type=str, default='data/AVVP/LLP_dataset/feats/r2plus1d_18/', help="video dir")
        parser.add_argument(
            "--label_train", type=str, default="data/AVVP/AVVP_train.csv", help="weak train csv file")
        parser.add_argument(
            "--label_val", type=str, default="data/AVVP/AVVP_val_pd.csv", help="weak val csv file")
        parser.add_argument(
            "--label_test", type=str, default="data/AVVP/AVVP_test_pd.csv", help="weak test csv file")
        parser.add_argument(
            '--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
        parser.add_argument(
            '--epochs', type=int, default=40, metavar='N', help='number of epochs to train (default: 40)')
        parser.add_argument(
            '--warmup_epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 2)')
        parser.add_argument(
            '--lr', type=float, default=3e-4, metavar='LR',  help='learning rate (default: 3e-4)')
        parser.add_argument(
            '--weight_decay', type=float, default=0, help='Weight Decay')
        parser.add_argument(
            "--model", type=str, default='MGN_Net', help="with model to use")
        parser.add_argument(
            "--mode", type=str, default='train', help="with mode to use")
        parser.add_argument(
            '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument(
            '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument(
            "--model_save_dir", type=str, default='models/', help="model save dir")
        parser.add_argument(
            "--checkpoint", type=str, default='MGN_Net_avmoe', help="save model name")
        parser.add_argument(
            '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu device number')
        
        # Model settings
        parser.add_argument(
            "--unimodal_assign", type=str, default='soft', help="type of unimodal grouping assignment")
        parser.add_argument(
            "--crossmodal_assign", type=str, default='soft', help="type of crossmodal grouping assignment")
        parser.add_argument(
            '--dim', type=int, default=128, help='dimensionality of features')
        parser.add_argument(
            '--depth_aud', type=int, default=3, help='depth of audio transformers')
        parser.add_argument(
            '--depth_vis', type=int, default=3, help='depth of visual transformers')
        parser.add_argument(
            '--depth_av', type=int, default=6, help='depth of audio-visual transformers')

        parser.add_argument(
            '--audio_folder', type=str, default="data/AVVP/LLP_dataset/audio", help="raw audio path")
        parser.add_argument(
            '--video_folder', type=str, default="data/AVVP/LLP_dataset/frame", help="video frame path")
        parser.add_argument(
            '--audio_length', type=float, default= 1, help='audio length')
        parser.add_argument(
            '--num_workers', type=int, default=16, help='worker for dataloader')
        parser.add_argument(
            '--model_name', type=str, default=None, help="for log")

        parser.add_argument(
            '--qkv_fusion', type=int, default=1, help="qkv fusion")

        parser.add_argument(
            '--adapter_kind', type=str, default='bottleneck', help="for log")

        parser.add_argument(
            '--start_tune_layers', type=int, default=0, help="tune top k")

        parser.add_argument(
            '--start_fusion_layers', type=int, default=0, help="tune top k")

        parser.add_argument(
            '--Adapter_downsample', type=int, default=8, help="tune top k")


        parser.add_argument(
            '--num_conv_group', type=int, default=2, help="group conv")
        parser.add_argument(
            '--is_audio_adapter_p1', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_audio_adapter_p2', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_audio_adapter_p3', type=int, default=1, help="TF audio adapter")

        parser.add_argument(
            '--is_bn', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_gate', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_multimodal', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_before_layernorm', type=int, default=1, help="TF audio adapter")
        parser.add_argument(
            '--is_post_layernorm', type=int, default=1, help="TF audio adapter")

        parser.add_argument(
            '--is_vit_ln', type=int, default=0, help="TF Vit")

        parser.add_argument(
            '--is_fusion_before', type=int, default=1, help="TF Vit")

        parser.add_argument(
            '--num_tokens', type=int, default=32, help="num of MBT tokens")

        parser.add_argument(
            '--vis_encoder_type', type=str, default="vit", help="type of visual backbone")
        parser.add_argument(
            '--backbone_type', type=str, default='audioset', help="the backbone of htsat")

        parser.add_argument(
            '--root_path', type=str, default=data_grandparent_dir)
        
        parser.add_argument(
            '--accum_itr', type=int, default=8, help='number of gradient iterations')
        
        # ly: new args
        parser.add_argument(
            '--is_router', type=int, default=1, help="whether use the router")
        parser.add_argument(
            '--num_multimodal_experts', type=int, default=1, help="number of the multimodal_experts")
        parser.add_argument(
            '--num_singlemodal_experts', type=int, default=1, help="number of the singlemodal_experts")
        parser.add_argument(
            '--wandb', type=int, default=0, help="whether use the wandb")
        parser.add_argument(
            '--checkpoint_path', type=str, default='/remote-home/yangli/DG-SCT/DG-SCT/checkpoints', help="path of checkpoint")
        parser.add_argument(
            '--is_load_checkpoint', type=int, default=0, help="whether load the best checkpoint")
        parser.add_argument(
            "--model_load_dir", type=str, default='models/', help="model load dir")
        parser.add_argument(
            "--use_load_balacing_loss", type=int, default=1, help="whether use load_balacing loss")


    args = parser.parse_args()

    if args.wandb:
        wandb.init(config=args, project="avvp", name=args.model_name)

    # ly: fix the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    # load model
    if args.model == 'MGN_Net':
        model = MGN_Net(args).to('cuda')
        if args.is_load_checkpoint:
            model.load_state_dict(torch.load(args.model_load_dir + args.checkpoint + ".pt"), strict=False)

        if args.mode == 'train' and len(args.gpu.split(',')) > 1:
            model = torch.nn.DataParallel(model).to('cuda')
        
    else:
        raise ('not recognized')
    
    # compute params
    param_group = []
    train_params = 0
    total_params = 0
    additional_params = 0
    for name, param in model.named_parameters():
        param.requires_grad = True
        tmp = 1
        for num in param.shape:
            tmp *= num

        if 'ViT' in name or 'swin' in name:
            if 'norm' in name:
                param.requires_grad = bool(args.is_vit_ln)
                total_params += tmp
                train_params += tmp
            else:
                param.requires_grad = False
                total_params += tmp
        elif 'htsat' in name:
            param.requires_grad = False
            total_params += tmp
            
        elif 'adapter_blocks' in name:
            param.requires_grad = True
            train_params += tmp
            additional_params += tmp
            total_params += tmp
            print('########### train layer:', name)
        elif 'temporal_attn' in name:
            param.requires_grad = True
            train_params += tmp
            additional_params += tmp
            total_params += tmp
        else:
            param.requires_grad = True
            train_params += tmp
            total_params += tmp
        
        param_group.append({"params": param, "lr":args.lr})
    print('####### Trainable params: %0.2f %% #######'%(train_params*100/total_params))
    print('####### Additional params: %0.2f %% #######'%(additional_params*100/(total_params-additional_params)))
    print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))
    
    if args.mode == 'train':
        train_dataset = LLP_dataset(args=args, 
                                    label=os.path.join(args.root_path, args.label_train), 
                                    audio_dir=os.path.join(args.root_path, args.audio_dir), 
                                    video_dir=os.path.join(args.root_path, args.video_dir), 
                                    st_dir=os.path.join(args.root_path, args.st_dir), 
                                    transform = transforms.Compose([ToTensor()]))
        val_dataset = LLP_dataset(args=args, 
                                  label=os.path.join(args.root_path, args.label_val), 
                                  audio_dir=os.path.join(args.root_path, args.audio_dir), 
                                  video_dir=os.path.join(args.root_path, args.video_dir), 
                                  st_dir=os.path.join(args.root_path, args.st_dir), 
                                  transform = transforms.Compose([ToTensor()]))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True, drop_last=True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        criterion = nn.BCELoss()
        criterion_ce = nn.CrossEntropyLoss()
        best_F = 0
        
        for epoch in range(0, args.epochs):
            train(args, model, train_loader, optimizer, criterion, criterion_ce, epoch)
            scheduler.step()
            F, F_event = eval(args, model, val_loader, os.path.join(args.root_path, args.label_val))

            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + args.model_name + str(best_F) + ".pt")
                print('#################### save model #####################')
                if args.wandb:
                    wandb.log({"avg_type val-best": F})

        # testing
        test_dataset = LLP_dataset(args=args, 
                                   label=os.path.join(args.root_path, args.label_test), 
                                   audio_dir=os.path.join(args.root_path, args.audio_dir), 
                                   video_dir=os.path.join(args.root_path, args.video_dir),  
                                   st_dir=os.path.join(args.root_path, args.st_dir), 
                                   transform = transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        model.load_state_dict(torch.load(args.model_load_dir + args.checkpoint + ".pt"))
        eval(args, model, test_loader, os.path.join(args.root_path, args.label_test))

    elif args.mode == 'val':
        test_dataset = LLP_dataset(args=args,
                                   label=os.path.join(args.root_path, args.label_val), 
                                   audio_dir=os.path.join(args.root_path, args.audio_dir), 
                                   video_dir=os.path.join(args.root_path, args.video_dir), 
                                   st_dir=os.path.join(args.root_path, args.st_dir), 
                                   transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        model.load_state_dict(torch.load(args.model_load_dir + args.checkpoint + ".pt"))
        eval(args, model, test_loader, os.path.join(args.root_path, args.label_val))
    else:
        test_dataset = LLP_dataset(args=args, 
                                   label=os.path.join(args.root_path, args.label_test), 
                                   audio_dir=os.path.join(args.root_path, args.audio_dir), 
                                   video_dir=os.path.join(args.root_path, args.video_dir),  
                                   st_dir=os.path.join(args.root_path, args.st_dir), 
                                   transform = transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(args.model_load_dir + args.checkpoint + ".pt"))
        eval(args, model, test_loader, os.path.join(args.root_path, args.label_test))

if __name__ == '__main__':
    main()