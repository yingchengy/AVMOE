gpu_to_use=0
model_save_dir=""

num_multimodal_experts=1
num_singlemodal_experts=2
num_tokens=2
skip_num=1
IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER=0
label_test_path=data/AVQA/json_update/avqa-test.json

python3 -u net_grd_avst/main_avst_v2.py --mode test \
	--audio_dir data/AVQA/vggish \
	--video_res14x14_dir data/AVQA/frames/ \
	--wandb 0 \
	--num_workers 16 \
	--batch-size 8 \
	--model_name swinv2_tune_av+vggish \
	--backbone_type audioset --Adapter_downsample 8 --num_tokens ${num_tokens} \
	--is_audio_adapter_p1 1 \
	--is_audio_adapter_p2 1 \
	--is_gate 0 \
	--avqa_fc_class 42 \
	--num_singlemodal_experts ${num_singlemodal_experts} \
	--num_multimodal_experts ${num_multimodal_experts} \
	--model_save_dir ${model_save_dir} \
	--num_skip ${skip_num} \
	--is_self_attention ${IS_SELF_ATTENTION_IN_SIGNLE_ADAPTER} \
	--label_test ${label_test_path} \
	--gpu ${gpu_to_use}

