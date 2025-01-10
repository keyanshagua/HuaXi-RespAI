#!/bin/bash
#SBATCH -J ViT_process  # 指定作业名
#SBATCH -p dzagnormal        # 指定队列
#SBATCH -N 1                 # 请求节点数
#SBATCH --ntasks-per-node=8  # 每个节点的任务数
#SBATCH -w gpu9
#SBATCH --cpus-per-task=1    # 每个任务的CPU核数
#SBATCH --gres=gpu:1         # 请求GPU数
#SBATCH -o %x.o%j            # 标准输出文件
#SBATCH -e %x.e%j            # 错误输出文件

# 加载必要的模块（如果有）并激活 conda 环境
source /work/home/ac609hjkef/software/miniconda3/bin/activate
conda activate torch1.10_dtk22.10_py3.8

python ViT_train.py

