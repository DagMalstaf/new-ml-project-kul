from setuptools import setup, Extension
from Cython.Build import cythonize
import os

extensions = [
    Extension(
        "mcts_simulation_cython", 
        ["mcts_simulation_cython.pyx"],
        language="c++",
        libraries=['pthread'],
        extra_compile_args=["-std=c++11", "-stdlib=libc++", "-Wno-nullability-completeness"],
    )
]

sdk_path = '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'
os.environ['CFLAGS'] = f'-isysroot {sdk_path}'
os.environ['LDFLAGS'] = f'-isysroot {sdk_path}'

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)

