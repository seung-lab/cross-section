import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++17', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++17', '-O3'
  ]

setuptools.setup(
  setup_requires=['pbr','pybind11','numpy','connected-components-3d'],
  cmdclass={"build_ext": build_ext},
  extras_require={},
  ext_modules=[
    Pybind11Extension(
        "fastxs3d",
        ["src/fastxs3d.cpp"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
  ],
  entry_points={},
  pbr=True
)
