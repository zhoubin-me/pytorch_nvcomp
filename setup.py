from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
nvcomp_root = os.environ.get("NVCOMP_ROOT", os.path.join(this_dir, "..", "nvcomp-cuda13"))
nvcomp_root = os.path.abspath(nvcomp_root)

include_dirs = [os.path.join(nvcomp_root, "include")]
library_dirs = [os.path.join(nvcomp_root, "lib")]

extra_nvcc_args = ["-O3"]
if os.environ.get("NVCC_ALLOW_UNSUPPORTED", "0") == "1":
    extra_nvcc_args.append("-allow-unsupported-compiler")

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": extra_nvcc_args,
}

setup(
    name="pytorch_nvcomp",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="pytorch_nvcomp._nvcomp_ext",
            sources=["pytorch_nvcomp/nvcomp_ext.cu"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["nvcomp"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
