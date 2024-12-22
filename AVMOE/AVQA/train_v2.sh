gpu_to_use=6

num_multimodal_experts=1
num_singlemodal_experts=2
is_init_from_checkpoint=1

model_load_dir=""

skip_num=1
LR=1e-4 
num_tokens=2 

IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=0 

GPU_TYPE="fast_3090"
VERSION=1
MODEL_SAVE_DIR="/data/anonymous/Code/AVMOE/AVMOEAVQA/net_grd_avst/avst_models/${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}_senum-${num_singlemodal_experts}_menum-${num_multimodal_experts}-lr-${LR}-skm-${skip_num}-numtok-${num_tokens}"
LOG_DIR="${GPU_TYPE}_out_v1_${gpu_to_use}_${VERSION}.log"

echo saving log to ${LOG_DIR}

echo saving model to ${MODEL_SAVE_DIR}

CUDA_VISIBLE_DEVICES=${gpu_to_use} nohup python3 -u net_grd_avst/main_avst_v2.py --mode train \
	--audio_dir data/AVQA/vggish \
	--video_res14x14_dir data/AVQA/frames/ \
	--wandb 0 \
	--num_workers 16 \
	--lr ${LR} \
	--batch-size 2 \
	--accum_itr=4 \
	--model_name swinv2_tune_av+vggish \
	--backbone_type audioset --Adapter_downsample 8 --num_tokens ${num_tokens} \
	--is_audio_adapter_p1 1 \
	--is_audio_adapter_p2 1 \
	--is_gate 0 \
	--avqa_fc_class 50 \
	--model_save_dir ${MODEL_SAVE_DIR}  \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--is_init_from_checkpoint ${is_init_from_checkpoint} \
	--model_load_dir ${model_load_dir} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
	--gpu ${gpu_to_use} > ${LOG_DIR} 2>&1 &


