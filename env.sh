#export TORCH_BLAS_PREFER_HIPBLASLT=0
#unset PYTORCH_HIP_ALLOC_CONF
# helps regain VRAM after crash
#export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:1024"

# fixed vram regain after OOM
#export AMD_SERIALIZE_KERNEL=1
#export USE_PYTORCH_KERNEL_CACHE=0
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:8

#[sandboxed]$ env | grep CUDA
#CUDA_VERSION=gfx1030
#CUDA_LAUNCH_BLOCKING=1
#PYTORCH_NO_CUDA_MEMORY_CACHING=1
#[sandboxed]$ env | grep TORCH
#TORCH_USE_HIP_DSA=1
#USE_PYTORCH_KERNEL_CACHE=0
#PYTORCH_HIP_ALLOC_CONF=roundup_power2_divisions:8
#PYTORCH_NO_CUDA_MEMORY_CACHING=1

unset HCC_AMDGPU_TARGET
unset PYTORCH_HIP_ALLOC_CONF CUDA_VERSION  HCC_AMDGPU_TARGET HIP_VISIBLE_DEVICES

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_BLAS_PREFER_HIPBLASLT=0

# fixes stupid perf issue I hope
export GTK_THEME=Adwaita:dark

# fixes top chunk corruption
#export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export LD_PRELOAD=/usr/lib/libjemalloc.so.2

# try debugging..
#export PYTORCH_COMPILE_DEBUG=1
#export TORCH_CUDNN_V8_API_DEBUG=1
#export PYTORCH_DEBUG_MPS_ALLOCATOR=1
#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_LOGS="+dynamo,graph_breaks,guards,recompiles,dynamic"

# fix OOM
# This one really does help, it made 1024x1024 training possible when otherwise
# there was just barely too much fragmentation. A value of 512 did NOT fix this issue.
#export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
# .. but 128 still gives me fragmentation on second run -> OOM
# .. 64 same issue.
#export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:256"
#export PYTORCH_HIP_ALLOC_CONF="max_non_split_rounding_mb:512" # still OOM
#export PYTORCH_HIP_ALLOC_CONF="backend:hipMallocAsync" # ERROR cause cuda < 11.X
#export MIOPEN_FIND_MODE=2
#export PYTORCH_TUNABLEOP_ENABLED=1 
#export PYTORCH_TUNABLEOP_VERBOSE=1
#export HSA_USE_UNIFIED_MEMORY=1
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.6"
#export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
#unset PYTORCH_HIP_ALLOC_CONF

# TO TRY for stability:
# - https://github.com/search?q=repo%3Acomfyanonymous%2FComfyUI%20os.environ&type=code
# - https://github.com/comfyanonymous/ComfyUI/issues/5759
#export MIOPEN_FIND_MODE=2
#export HSA_USE_UNIFIED_MEMORY=1
#export PYTORCH_TUNABLEOP_ENABLED=1 
#export PYTORCH_TUNABLEOP_VERBOSE=1
#export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF,backend:cudaMallocAsync"
#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#export HIP_VISIBLE_DEVICES="cuda:0"
#export CUDA_VISIBLE_DEVICES="cuda:0"

# needed for bitsandbytes / pytorch bug: https://github.com/pytorch/pytorch/issues/60477
export MIOPEN_USER_DB_PATH="/optane/miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
#rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
