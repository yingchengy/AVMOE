gpu_to_use=7

setting='S4'
visual_backbone="pvt" # "resnet" or "pvt"
        
num_multimodal_experts=1
num_singlemodal_experts=1
skip_num=1
accum_itr=4
# num_conv_group=2 
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=1 

GPU_TYPE="fast_3090"
VERSION=4

TRAINING_LOG_DIR=${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}.log
SAVING_LOG_DIR="Code/AVMOE/AVMOEAVS/models/${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}-senum-${num_singlemodal_experts}_menum-${num_multimodal_experts}"
# log_dir './train_logs' 
echo saving log to ${TRAINING_LOG_DIR}

echo saving model to ${SAVING_LOG_DIR}

CUDA_VISIBLE_DEVICES=${gpu_to_use} nohup python3 -u  train_v2.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 2 --num_tokens 32 --Adapter_downsample 8 \
        --max_epoches 50 \
        --accum_itr ${accum_itr} \
        --lr 0.0003 \
        --tpavi_stages 0 1 2 3 \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
        --wandb 0 \
        --log_dir ${SAVING_LOG_DIR} \
        --model_name s4-swinv2-tune-av  > ${TRAINING_LOG_DIR} 2>&1 &

