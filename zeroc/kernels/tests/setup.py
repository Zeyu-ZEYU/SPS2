from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zeyu_cuda_tests",
    ext_modules=[
        CUDAExtension(
            "zeyu_cuda_tests",
            [
                "zeyu_cuda_tests.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
