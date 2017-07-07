from __future__ import unicode_literals

import os
import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension

try:
    import cython
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    print("Error: cython package couldn't be found." +
          " Please, install it so I can proceed.")
    sys.exit(1)


def plink_extension():
    curdir = os.path.abspath(os.path.dirname(__file__))

    plink_folder = os.path.join(curdir, 'limix_ext/gcta/core/plink_/')

    src = ['write.pyx']
    src = [os.path.join(plink_folder, s) for s in src]

    hdr = ['write.pxd']
    hdr = [os.path.join(plink_folder, h) for h in hdr]

    depends = src + hdr

    ext = Extension('limix_ext/gcta/core/plink_/write', src, depends=depends)

    return ext


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner>=2.9'] if needs_pytest else []

    setup_requires = ['cffi>=1.7'] + pytest_runner
    install_requires = ['scipy-sugar>=1.0.1', 'pandas', 'fastlmm>=0.3']
    tests_require = install_requires

    metadata = dict(
        name='limix_ext',
        version='1.0.3',
        maintainer="Limix Developers",
        maintainer_email="horta@ebi.ac.uk",
        license="MIT",
        url='https://github.com/Horta/limix-ext',
        packages=find_packages(),
        zip_safe=False,
        ext_modules=cythonize([plink_extension()]),
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        include_package_data=True)

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
