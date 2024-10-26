#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -cwd
#$ -p -400

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/8.3.1 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 openjdk/11.0.22.0.7 nccl/2.14/2.14.3-1
conda activate colbert

# export TORCH_CUDA_ARCH_LIST="7.0"
# export NCCL_BLOCKING_WAIT=1
export PYTHONPATH=/home/ace14788tj/aken12_2/RAGatouille:$PYTHONPATH
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True

nvcc --version

task=clueweb09
corpus=clueweb09
prefix=q2e
topk=1000
model_name=colbert-ir/colbertv2.0
doc_max_length=256

CUDA_VISIBLE_DEVICES="" python retrieve.py \
 --task $task \
 --rewrite $prefix \
 --model_name colbert-ir/colbertv2.0 \
 --topk $topk --corpus $corpus