#!/usr/local_rwth/bin/zsh
 
# name the job
#SBATCH --job-name=cycleGAN_test

# request CPU resources
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G

# request GPU
#SBATCH --gres=gpu:1

# time limit
#SBATCH --time=0-06:00:00  

# specify account
#SBATCH --account=rwth1299

### beginning of executable commands
module load CUDA/11.8.0

$HOME/miniconda3/envs/GAN/bin/python $HOME/Thesis/code/ct-pore-analysis/code/cyclegan_gen.py
