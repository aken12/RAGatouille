source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/8.5.0 cuda/11.8/11.8.0 cudnn/8.8/8.8.1 openjdk/11.0.22.0.7 nccl/2.14/2.14.3-1
conda activate colbert

export TORCH_CUDA_ARCH_LIST="7.0"
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_TIMEOUT=1000
export PYTHONPATH=/home/ace14788tj/aken12_2/RAGatouille:$PYTHONPATH
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True

nvcc --version

task=msmarco
prefix=default
topk=1000
model_name=colbert-ir/colbertv2.0
doc_max_length=256

CUDA_VISIBLE_DEVICES="0,1,2,3" python index.py \
 --task $task \
 --rewrite $prefix \
 --model_name $model_name \
 --doc_max_length $doc_max_length

 # qrsh -g grpname -l rt_F=1 -l h_rt=1:00:00
 