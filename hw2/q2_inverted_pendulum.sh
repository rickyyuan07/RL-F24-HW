#!/bin/bash
# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.001 -rtg \
# --exp_name q2_b50000_r0.001

# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.005 -rtg \
# --exp_name q2_b50000_r0.005

# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 50000 -lr 0.01 -rtg \
# --exp_name q2_b50000_r0.01

# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.001 -rtg \
# --exp_name q2_b10000_r0.001

# python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
# --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 10000 -lr 0.005 -rtg \
# --exp_name q2_b10000_r0.005

python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 0.02 -rtg \
--exp_name q2_b1000_r0.02