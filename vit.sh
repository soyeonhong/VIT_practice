#!/bin/bash

#SBATCH --job-name vit

#SBATCH --gres=gpu:1

#SBATCH -p batch

#SBATCH --cpus-per-gpu=4

#SBATCH --mem=20gb

#SBATCH --time 1-0

#SBATCH -o /data/soyeon/vit/slurm/logs/slurm-%A_%x.out

conda activate vit

python /data/soyeon/vit/main.py

exit 0