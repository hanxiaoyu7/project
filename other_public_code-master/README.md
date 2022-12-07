# other_public_code

## collect_env.py
文件来自```detectron2\detectron2\utils\```可以测试Pytorch环境
取出来后可以直接执行（但是可能需要额外安装一些里面用到的库）
执行后输出如下：
----------------------  ---------------------------------------------------------------------------------
sys.platform            win32
Python                  3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
numpy                   1.18.5
detectron2              failed to import
detectron2._C           not built correctly: No module named 'detectron2'
DETECTRON2_ENV_MODULE   <not set>
PyTorch                 1.7.1 @C:\Users\li\anaconda3\lib\site-packages\torch
PyTorch debug build     False
GPU available           True
GPU 0                   GeForce 920M (arch=3.5)
CUDA_HOME               C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
Pillow                  7.2.0
torchvision             0.8.2 @C:\Users\li\anaconda3\lib\site-packages\torchvision
torchvision arch flags  C:\Users\li\anaconda3\lib\site-packages\torchvision\_C.pyd; cannot find cuobjdump
fvcore                  0.1.5.post20210410
iopath                  0.1.7
cv2                     Not found
----------------------  ---------------------------------------------------------------------------------
PyTorch built with:
  - C++ Version: 199711
  - MSVC 192729112
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191125 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v1.6.0 (Git Hash 5ef631a030a6f73131c77892041042805a06064f)
  - OpenMP 2019
  - CPU capability usage: AVX2
  - CUDA Runtime 10.1
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_37,code=compute_37
  - CuDNN 7.6.4
  - Magma 2.5.4
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /w /bigobj -openmp:experimental -DNDEBUG -DUSE_FBGEMM -DUSE_VULKAN_WRAPPER, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=OFF, USE_OPENMP=ON,
