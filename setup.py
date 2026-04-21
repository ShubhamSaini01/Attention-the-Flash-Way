from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="flash_attention_ext",
    ext_modules=[
        CUDAExtension(
            name="flash_attention_ext",
            sources=[
                "attention/attention_binding.cpp",
                "attention/attention_naive.cu",
                "attention/attention_optimized.cu",
                "attention/attention_flash.cu",
                "attention/attention_flash_v2.cu",
                "attention/attention_flash_v3.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
