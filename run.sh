export CUDA_VISIBLE_DEVICES=0,1,2
numactl --cpunodebind=0 --membind=0 torchrun --nproc_per_node=3 train.py
