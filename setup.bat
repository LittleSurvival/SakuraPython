@echo off

:: Change directory to /kaggle/working
cd /kaggle/working

:: Remove llama.cpp directory if it exists
rmdir /s /q llama.cpp

:: Echo preparation message
echo 准备编译llama.cpp...

:: Clone llama.cpp repository
git clone -c advice.detachedHead=false -b b3855 --depth 1 https://github.com/ggerganov/llama.cpp.git

:: Copy CUDA targets
xcopy /s /e /y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\targets" "C:\Program Files\NVIDIA Corporation"

:: Echo compilation message
echo 开始编译llama.cpp...

:: Change directory to the cloned llama.cpp
cd llama.cpp

:: Compile llama.cpp with CUDA support
set NUM_PROC=NUMBER_OF_PROCESSORS
make LLAMA_CUDA=1 CUDA_PATH="C:\Program Files\NVIDIA Corporation" server -j%NUM_PROC%

:: List files (Windows equivalent of `ls -lh`)
dir /s /b server

:: Echo Python environment setup message
echo 配置python环境...

:: Install pyngrok using pip
pip install pyngrok

:: Echo compilation complete
echo 编译完成