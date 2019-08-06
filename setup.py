import torch
import os
import glob
from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
from torch.utils.cpp_extension import (
    CUDA_HOME, 
    CUDAExtension,
    CppExtension
)

requirements = (
    "torchvision",
    "ninja",
    "yacs",
    "cython",
    "matplotlib",
    "tqdm",
    "opencv-python",
    "scikit-image"
)

def get_extension():
    extensions_dir = os.path.join("maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cpu

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        ),
        extension(
            name='toolkit.utils.region',
            sources=[
                'toolkit/utils/region.pyx',
                'toolkit/utils/src/region.c',
            ],
            include_dirs=[
                'toolkit/utils/src'
            ]
        )
    ]

    return ext_modules

# ext_modules = [
#     Extension(
#         name='toolkit.utils.region',
#         sources=[
#             'toolkit/utils/region.pyx',
#             'toolkit/utils/src/region.c',
#         ],
#         include_dirs=[
#             'toolkit/utils/src'
#         ]
#     )
# ]

# setup(
#     name='toolkit',
#     packages=['toolkit'],
#     ext_modules=cythonize(ext_modules)
# )


setup(
    name='toolkit',
    packages=['toolkit'],
    ext_modules=get_extension(),
    install_requires=requirements,
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    include_package_data=True
)
