#!/bin/bash
### Stage1 Training Script
### Recommend using 32 A100 cluster training takes approximately 1.5 days, but you can also use 8 A100 with more time

# Basic Configuration
# Adjusted for single 96GB GPU
BATCH_SIZE=10
EPOCHS=10
GAC=1
LR=1e-4

# Training Parameters
TRAIN_ARGS="--output_dir=./outputs \
--name=postermaker_debug_stage1 \
--mixed_precision=fp16 \
--learning_rate=${LR} \
--num_train_epochs=${EPOCHS} \
--train_batch_size=${BATCH_SIZE} \
--gradient_accumulation_steps=${GAC} \
--lr_warmup_steps=2000 \
--lr_scheduler=constant_with_warmup \
--adam_epsilon=1e-15 \
--dataloader_num_workers=10 \
--checkpointing_steps=5000 \
--validation_steps=1 \
--resolution_h=1024 \
--resolution_w=1024 \
--pretrained_model_name_or_path=../checkpoints/stable-diffusion-3-medium-diffusers/ \
--ctrl_layers=12 \
--max_num_texts=7 \
--char_padding_to_len=16 \
--text_feature_drop=0.1 \
--p_drop_caption=0 \
--cfg_scale=5.0 \
--gradient_checkpointing"

# Start Training
echo "Starting Stage1 Training..."
echo "Training will automatically perform validation every ${validation_steps} steps"
HF_HUB_OFFLINE=1 WORLD_SIZE=1 RANK=-1 python3 train_sd3_stage1.py $TRAIN_ARGS

echo "Stage1 Training Completed!"




