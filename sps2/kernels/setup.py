from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="zeroc_cuda",
    ext_modules=[
        CUDAExtension(
            "zc_blas",
            [
                "zc_blas.cu",
            ],
        ),
        CUDAExtension(
            "zc_bmm_half",
            [
                "zc_bmm_half.cu",
            ],
        ),
        CUDAExtension(
            "zc_bmm_uint8",
            [
                "zc_bmm_uint8.cu",
            ],
        ),
        CUDAExtension(
            "zc_softmax",
            [
                "zc_softmax.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
