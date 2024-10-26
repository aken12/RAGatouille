#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=60:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/8.5.0 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 openjdk/11.0.22.0.7 nccl/2.14/2.14.3-1
conda activate colbert_v2

# export TORCH_CUDA_ARCH_LIST="7.0"
# export NCCL_BLOCKING_WAIT=1
export PYTHONPATH=/home/ace14788tj/aken12_2/RAGatouille:$PYTHONPATH
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True

nvcc --version

task=clueweb09
prefix=default
topk=1000
model_name=colbert-ir/colbertv2.0
doc_max_length=256

CUDA_VISIBLE_DEVICES="0,1,2,3" python index.py \
 --task $task \
 --rewrite $prefix \
 --model_name $model_name \
 --doc_max_length $doc_max_length

 ### conda install -c pytorch -c nvidia faiss-gpu=1.8.0
 ### /home/ace14788tj/extdisk/.pyenv/versions/miniforge3-23.11.0-0/lib/python3.10/site-packages/torch/include/ATen/core/boxing/impl/boxing.h