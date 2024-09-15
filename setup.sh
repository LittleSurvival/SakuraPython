%cd -q /kaggle/working
!rm -rf ./llama.cpp

!echo 准备编译llama.cpp...
!git clone -q -c advice.detachedHead=false -b b2859 --depth 1 https://github.com/ggerganov/llama.cpp.git
!cp -r /usr/local/cuda-12.1/targets /usr/local/nvidia/

!echo 开始编译llama.cpp...
%cd -q /kaggle/working/llama.cpp/
!make LLAMA_CUDA=1 CUDA_PATH=/usr/local/nvidia server -j$(nproc) -s
!ls -lh ./server

!echo 配置python环境...
!pip install -q pyngrok

!echo 编译完成
