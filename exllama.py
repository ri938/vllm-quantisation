import os

from torch.utils.cpp_extension import load
import torch

extension_name = "exllama_ext"
library_dir = os.path.dirname(os.path.abspath(__file__))
verbose = False
windows = False


exllama_ext = load(
    name = extension_name,
    sources = [
        os.path.join(library_dir, "exllama_ext/exllama_ext.cpp"),
        os.path.join(library_dir, "exllama_ext/cuda_buffers.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_matrix.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/column_remap.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rms_norm.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rope.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/half_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_attn.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_mlp.cu"),
        os.path.join(library_dir, "exllama_ext/cpu_func/rep_penalty.cpp")
    ],
    extra_include_paths = [os.path.join(library_dir, "exllama_ext")],
    verbose = verbose,
    extra_ldflags = (["cublas.lib"] + ([f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"] if sys.base_prefix != sys.prefix else [])) if windows else [],
    extra_cuda_cflags = ["-lineinfo"] + (["-U__HIP_NO_HALF_CONVERSIONS__", "-O3"] if torch.version.hip else []),
    extra_cflags = ["-O3"]
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
)


from exllama_ext import q4_matmul
from exllama_ext import make_q4
