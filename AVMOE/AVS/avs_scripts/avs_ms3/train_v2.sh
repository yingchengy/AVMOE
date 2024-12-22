gpu_to_use=5

setting='MS3'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"

num_multimodal_experts=2
num_singlemodal_experts=2
skip_num=2
use_load_balacing_loss=1
load_balancing_loss_weight=0.01
accum_itr=1
# num_conv_group=2 
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=1 
self_attention_version="v2"

GPU_TYPE="fast_3090"
VERSION=9

TRAINING_LOG_DIR=${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}.log
SAVING_LOG_DIR="/data/anonymous/Code/AVMOE/AVMOEAVS/models/${setting}/${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}-senum-${num_singlemodal_experts}_menum-${num_multimodal_experts}"

echo saving log to ${TRAINING_LOG_DIR}

echo saving model to ${SAVING_LOG_DIR}

CUDA_VISIBLE_DEVICES=${gpu_to_use} nohup python3 -u  train_v2.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --max_epoches 60 \
        --train_batch_size 2 \
        --accum_itr=${accum_itr} \
        --lr 0.00015 \
        --tpavi_stages 0 1 2 3 \
        --tpavi_va_flag 1 \
        --masked_av_flag \
        --masked_av_stages 0 1 2 3 \
        --lambda_1 0.5 \
        --kl_flag \
        --sa_loss_flag \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
        --self_attention_version ${self_attention_version} \
        --use_load_balacing_loss ${use_load_balacing_loss} \
        --load_balancing_loss_weight ${load_balancing_loss_weight} \
        --wandb 0 \
        --log_dir ${SAVING_LOG_DIR} \
        --gpu ${gpu_to_use}  > ${TRAINING_LOG_DIR} 2>&1 &