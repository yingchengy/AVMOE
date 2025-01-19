python main.py --mode train \
    --model_save_dir models/ \
    --unimodal_assign soft --crossmodal_assign soft \
    --epochs 40 --gpu='0' --seed 23 --wandb 0 \
    --depth_aud 3 --depth_vis 3 --depth_av 6 \
    --Adapter_downsample 8 --batch_size 1 --accum_itr 8 \
    --is_audio_adapter_p1 1 --is_audio_adapter_p2 1 --is_audio_adapter_p3 0 \
    --is_before_layernorm 1 --is_bn 0 --is_fusion_before 1 --is_gate 1  --is_post_layernorm 1 --is_vit_ln 0 \
    --num_conv_group 2 --num_tokens 32 --num_workers 16 \
    --num_multimodal_experts 1 --num_singlemodal_experts 1 --is_router 1 \
    --is_load_checkpoint 0 --use_load_balacing_loss 1

# --batch_size 8