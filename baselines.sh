#! /bin/zsh

python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b128-lr5e4-wrm17.pt --valid_path=/data/encoded_audioset/balanced --warmup 17 --lr 0.0005 --epochs 200 --num_workers 0 --batch_size 256
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b128-lr5e4-wrm34.pt --valid_path=/data/encoded_audioset/balanced --warmup 34 --lr 0.0005 --epochs 200 --num_workers 0 --batch_size 256
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b128-lr1e4-wrm17.pt --valid_path=/data/encoded_audioset/balanced --warmup 17 --lr 0.0001 --epochs 200 --num_workers 0 --batch_size 256
python train_baseline.py -f /data/encoded_audioset/unbalanced -o models/baseline-1.7M-large-b128-lr1e4-wrm34.pt --valid_path=/data/encoded_audioset/balanced --warmup 34 --lr 0.0001 --epochs 200 --num_workers 0 --batch_size 256
