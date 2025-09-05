#!/bin/sh
# Options SBATCH :

#SBATCH --job-name=regtest
#SBATCH --cpus-per-task=4

#SBATCH --mail-type=END
#SBATCH --mail-user=matthis.bernardini@etu.inp-toulouse.fr

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1

HEAD="srun singularity exec /apps/containerCollections/CUDA12/pytorch2-NGC-24-02.sif $HOME/GANs-Textures-Generation/SGAN/env_SGAN/bin/python"
cd $HOME/GANs-Textures-Generation/SGAN/
$HEAD jobW.py --textureName "barca.jpg" --epoch 20001 --netDepth 5 --multD 1 --latentSize 4 --latentCanal 20 --sampleLatentSize 16 --weightDecay 0 --plotRegLosses True


