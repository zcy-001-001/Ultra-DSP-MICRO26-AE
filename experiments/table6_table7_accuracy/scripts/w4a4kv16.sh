#!/usr/bin/sh
torchrun --nnodes 1 --nproc_per_node 8 --master-addr localhost --master-port 8902 main.py \
 --output_dir output/llama2_w4a4kv16 --model weights/Llama-2-7b-hf  \
 --loss_type=kl_top --post_attn=True \
 --rotate_ov=True --rotate_post_rope=False --online_qk_hadamard=False --smooth_qk=True --smooth_ov=True --smooth_up_down=True --smooth_norm_linear=True --bf16=True --lm_eval=True --per_device_train_batch_size=4 \
 --max_steps=100 --a_bits=4 --v_bits=16 --k_bits=16 --down_bits=4 \
 --train_enable_wquant=False --sub_mean False  --distribute=True --use_klt