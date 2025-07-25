#!/bin/sh
# Options SBATCH :

#SBATCH --job-name=SGAN5-GPU
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=END
#SBATCH --mail-user=matthis.bernardini@etu.inp-toulouse.fr

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1

HEAD="srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif $HOME/GANs-Textures-Generation/SGAN/env_SGAN/bin/python"
cd $HOME/GANs-Textures-Generation/SGAN/
$HEAD job.py --textureName "barca.jpg" --epoch 20001 --netDepth 4 --patchSize 64

