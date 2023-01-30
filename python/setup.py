import os
import shutil
import io
import sys
from setuptools import setup, find_packages
from subprocess import check_output
from setuptools.dist import Distribution
from platform import system

if "--universal" in sys.argv:
  raise ValueError("Creating py2.py3 wheels is not supported")

CURRENT_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

BUILD_DIR = "../build/lib/"

libname = 'libdlr.so'
if sys.platform == 'win32':
  libname = 'dlr.dll'
elif sys.platform == 'darwin':
  libname = 'libdlr.dylib'

LIB_PATH = os.path.join(BUILD_DIR, libname)

wheel_include_libs = False
if os.path.exists(LIB_PATH):
  print("Found", libname, "at", LIB_PATH)
  include_package_data = True
  data_files = [('dlr', [LIB_PATH,])]
  if "bdist_wheel" in sys.argv:
    wheel_include_libs = True
    data_files = None
else:
  print(libname, "is not found!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print("!!! Preparing universal py3 version of DLR !!!")
  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  include_package_data = False
  data_files = None

# For bdist_wheel only
if wheel_include_libs:
  with open("MANIFEST.in", "w") as fo:
    shutil.copy(LIB_PATH, os.path.join(CURRENT_DIR, "dlr"))
    shutil.copytree(os.path.join(CURRENT_DIR, "../include"),
                    os.path.join(CURRENT_DIR, "dlr/include"))
    shutil.copytree(os.path.join(CURRENT_DIR, "../3rdparty/tvm/3rdparty/dlpack/include/dlpack"),
                    os.path.join(CURRENT_DIR, "dlr/include/dlpack"))
    fo.write("include dlr/%s\n" % libname)
    fo.write("recursive-include dlr/include *\n")

# fetch meta data
METADATA_PY = os.path.abspath("./dlr/metadata.py")
METADATA_PATH = {"__file__": METADATA_PY}
METADATA_BIN = open(METADATA_PY, "rb")
exec(compile(METADATA_BIN.read(), METADATA_PY, 'exec'), METADATA_PATH, METADATA_PATH)
METADATA_BIN.close()

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

setup(
    name="dlr",
    version=METADATA_PATH['VERSION'],

    zip_safe=False,
    install_requires=['numpy', 'requests', "distro"],

    # declare your packages
    packages=find_packages(),

    # include data files
    include_package_data=include_package_data,
    data_files=data_files,

    description = 'Common runtime for machine learning models compiled by \
        AWS SageMaker Neo, TVM, or TreeLite.',
    long_description=io.open(os.path.join(CURRENT_DIR, '../README_TI.md'), encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/TexasInstruments/neo-ai-dlr',
    license = "Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires = '>=3.5',
    #distclass=BinaryDistribution,
)

if wheel_include_libs:
  # Wheel cleanup
  os.remove("MANIFEST.in")
  os.remove("dlr/%s" % libname)
  shutil.rmtree(os.path.join(CURRENT_DIR, "dlr/include"))
