import os
import sysconfig
from setuptools import setup

# The Python running this build is a standalone build compiled with GCC 14, so
# its sysconfig CFLAGS embed `-fdebug-default-version=4` — a flag system
# compilers older than GCC 14 (e.g. Ubuntu 24.04's g++-13) reject. Strip it
# (and the obsolete -Wstrict-prototypes) before setuptools/torch read the
# config for the extension build.
_STRIP_PREFIXES = ('-fdebug-default-version', '-Wstrict-prototypes')


def _strip_cflags(config_vars):
    for var in ('CFLAGS', 'OPT'):
        flags = config_vars.get(var)
        if flags:
            config_vars[var] = ' '.join(
                flag for flag in flags.split()
                if not flag.startswith(_STRIP_PREFIXES)
            )


_strip_cflags(sysconfig.get_config_vars())

# setuptools' vendored distutils caches its own copy of the config vars on
# first use; patch that cache too in case it was already populated.
import distutils.sysconfig as distutils_sysconfig
_strip_cflags(distutils_sysconfig.get_config_vars())

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src = 'src'
# Empty sources are skipped: the vendored tree carries a 0-byte
# src/knnquery/knnquery_cuda.cu whose object name collides with its
# sibling knnquery_cuda.cpp, which ninja rejects ("multiple rules
# generate ... knnquery_cuda.o"). It contributes no symbols — the real
# kernels live in knnquery_cuda_kernel.cu.
sources = [path for root, dirs, files in os.walk(src) for file in files
           if (file.endswith('.cpp') or file.endswith('.cu'))
           and os.path.getsize(path := os.path.join(root, file)) > 0]

setup(
    name='pointops2',
    version='1.0',
    install_requires=["torch", "numpy"],
    packages=["pointops2"],
    package_dir={"pointops2": "functions"},
    ext_modules=[
        CUDAExtension(
            name='pointops2_cuda',
            sources=sources,
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
