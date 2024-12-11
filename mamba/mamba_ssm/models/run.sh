WANDB_PROJECT=vetmamba python3 train.py \
--model_name_or_path <path-to-storing-your-model> \
--train_file <path-to-train-dataset> \
--block_size 4096 \
--num_workers 20 \
--checkpoint_steps 100
