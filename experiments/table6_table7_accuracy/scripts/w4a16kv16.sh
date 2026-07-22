#!/usr/bin/sh
torchrun --nnodes 1 --nproc_per_node 4 --master-addr localhost --master-port 8901 main.py \
 --output_dir output/llama2w4a16kv16 --model weights/Llama-2-7b-hf  \
 --loss_type=kl_top --post_attn=True \
 --rotate_ov=True --rotate_post_rope=False --online_qk_hadamard=False --smooth_qk=True --smooth_ov=True --smooth_up_down=True --smooth_norm_linear=True --bf16=True --lm_eval=True --per_device_train_batch_size=4 \
 --max_steps=100 --a_bits=16 --v_bits=16 --k_bits=16 --down_bits=16 --w_clip=True\
 --train_enable_wquant=True --sub_mean False --distribute=True
