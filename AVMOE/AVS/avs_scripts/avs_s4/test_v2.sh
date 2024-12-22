gpu_to_use=3

setting='S4'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"

model_checkpoint_dir=""
num_multimodal_experts=2
num_singlemodal_experts=2
skip_num=2
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=1 
self_attention_version="v2"
visual_only=0

CUDA_VISIBLE_DEVICES=${gpu_to_use} python test_v2.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights ${model_checkpoint_dir} \
    --num_tokens 32 --Adapter_downsample 8 \
    --test_batch_size 1 \
    --tpavi_va_flag 1 \
    --tpavi_stages 0 1 2 3 \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
    --visual_only ${visual_only} \
    --save_pred_mask