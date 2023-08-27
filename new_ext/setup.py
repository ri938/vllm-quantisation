# adapted from llm-awq: https://github.com/mit-han-lab/llm-awq

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17"],
    #"nvcc": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "-std=c++17", "-lineinfo"],
}

setup(
    name="new_inf",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="new_inf",
            sources=[
                "new_kernels/pybind.cpp", 
                "new_kernels/gemm_cuda_gen.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
