
gpu_to_use=3

setting='MS3'
visual_backbone="pvt" # "resnet" or "pvt"

num_multimodal_experts=2
num_singlemodal_experts=2
skip_num=2
accum_itr=1
# num_conv_group=2 
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=1 
self_attention_version="v2"
visual_only=0

CUDA_VISIBLE_DEVICES=${gpu_to_use} python3 -u  test_v2.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "MS3_pvt_best.pth" \
    --test_batch_size 1 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag  1 \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
    --self_attention_version ${self_attention_version} \
    --visual_only ${visual_only} \
    --save_pred_mask

