gpu_to_use=0
model_size=large
is_init_best_checkpoint=1
is_inter_in_cmbs=0
is_cmbs=1
is_temporal_att=1
num_multimodal_experts=1
num_singlemodal_experts=1
skip_num=2
accum_itr=8
is_gate=1 
is_bn=1
is_post_layernorm=1
num_conv_group=2
Adapter_downsample=8 
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=0
is_audio_adapter_p1=1
is_audio_adapter_p2=1

decay=0.35
decay_epoch=3

GPU_TYPE="fast_3090"
VERSION=3
MODEL_SAVE_DIR="/data/anonymous/Code/AVMOE/AVMOEAVE/models/${GPU_TYPE}_moe_v1_${gpu_to_use}_${VERSION}_${model_size}-initbest-${is_init_best_checkpoint}-iscmbs-${is_cmbs}-isintercmbs-${is_inter_in_cmbs}-istempa-${is_temporal_att}-skipnum-${skip_num}-senum-${num_singlemodal_experts}_menum-${num_multimodal_experts}-accum-${accum_itr}"
LOG_DIR=${GPU_TYPE}_out_moe_v1_${gpu_to_use}_${VERSION}.log

echo saving log to ${LOG_DIR}

echo saving model to ${MODEL_SAVE_DIR}

CUDA_VISIBLE_DEVICES=${gpu_to_use} nohup python3 -u main_trans_v3.py --Adapter_downsample=${Adapter_downsample} --accum_itr=${accum_itr} \
	--batch_size=2 --decay=${decay} --decay_epoch=${decay_epoch} --early_stop=20 --epochs=50 --is_audio_adapter_p1=${is_audio_adapter_p1} --is_audio_adapter_p2=${is_audio_adapter_p2} \
	--is_audio_adapter_p3=0 --is_before_layernorm=1 --is_bn=${is_bn} --is_fusion_before=1 --is_gate=${is_gate}  \
	--is_post_layernorm=${is_post_layernorm} --is_vit_ln=0 --lr=5e-04 --lr_mlp=5e-06 --mode=train \
	--model=MMIL_Net --num_conv_group=${num_conv_group} --num_tokens=32 --num_workers=16 --seed 999 \
	--backbone_type audioset \
  --model_size ${model_size} \
  --is_init_best_checkpoint ${is_init_best_checkpoint} \
	--is_inter_in_cmbs ${is_inter_in_cmbs} \
	--is_cmbs ${is_cmbs} \
	--is_temporal_att ${is_temporal_att} \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
  --model_save_dir ${MODEL_SAVE_DIR}  \
	--gpu ${gpu_to_use} > ${LOG_DIR} 2>&1 &
