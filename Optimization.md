export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
/usr/local/cuda/lib

export OPENMM_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3
/opt/CUDA/cuda-6.0/bin/nvcc

python -m simtk.testInstallation
