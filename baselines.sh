#! /bin/zsh

python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a0.5.pt --valid_path /data/encoded_audioset/balanced --alpha 0.50 --epochs 200 --num_workers 8 --batch_size 64 --wandb
# python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a0.5.pt --valid_path /data/encoded_audioset/balanced --alpha 0.75 --epochs 200 --num_workers 8 --batch_size 64 --wandb
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a1.0.pt --valid_path /data/encoded_audioset/balanced --alpha 1.00 --epochs 200 --num_workers 8 --batch_size 64 --wandb
# python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a0.5.pt --valid_path /data/encoded_audioset/balanced --alpha 1.50 --epochs 200 --num_workers 8 --batch_size 64 --wandb
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a2.0.pt --valid_path /data/encoded_audioset/balanced --alpha 2.00 --epochs 200 --num_workers 8 --batch_size 64 --wandb
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b256-a4.0.pt --valid_path /data/encoded_audioset/balanced --alpha 4.00 --epochs 200 --num_workers 8 --batch_size 64 --wandb
