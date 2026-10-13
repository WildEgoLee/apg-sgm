import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ROOT = os.path.abspath(os.path.dirname(__file__))

sources = [
    "python/src/binding.cpp",
    "src/ablation.cpp",
    "src/c_api.cpp",
    "src/confidence_estimator.cpp",
    "src/config.cpp",
    "src/cost_aggregator.cpp",
    "src/cost_aggregator_avx2.cpp",
    "src/cost_computer.cpp",
    "src/cost_computer_avx2.cpp",
    "src/image.cpp",
    "src/metrics.cpp",
    "src/pipeline.cpp",
    "src/postprocess.cpp",
    "src/prior_estimator.cpp",
    "src/refiner.cpp",
    "src/refiner_avx2.cpp",
    "src/sgm_optimizer.cpp",
    "src/sgm_optimizer_avx2.cpp",
]

include_dirs = [
    os.path.join(ROOT, "include"),
]

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/arch:AVX2", "/openmp", "/O2", "/EHsc", "/std:c++17"]
    extra_link_args = []
else:
    extra_compile_args = ["-mavx2", "-fopenmp", "-O3", "-std=c++17"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "apg_sgm._core",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    packages=["apg_sgm"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
