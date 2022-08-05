# TI's fork of DLR

  TI's DLR fork adds the capability of running TVM deployable module
compiled by [TI's TVM fork](https://github.com/TexasInstruments/tvm/tree/tidl-j7)
that supports TVM+TIDL flow.


## Branches

  * tidl-j7 - This is the release branch
    (in sync with TI's TVM tidl-j7 branch)
  * tidl-j7-dev - This is the internal development branch


## Tags/Releases

  TI's DLR releases are synchronized with TI's PSDK (Processor SDK) release.
The following are the tags compatible with the target file system in PSDK
releases.

| PSDK release | TVM+TIDL release tag | Key features                                              |
|--------------|----------------------|-----------------------------------------------------------|
| 8.4          | TIDL\_PSDK\_8.4      | update tvm                                                |
| 8.2          | TIDL\_PSDK\_8.2, TI.8.2.0 | merged with neo-ai-dlr v1.10.0                       |
| 8.1          | TIDL\_PSDK\_8.0      |                                                           |
| 8.0          | TIDL\_PSDK\_8.0      |                                                           |
| 7.3          | TIDL\_PSDK\_7.3      | merged with neo-ai-dlr v1.8.0                             |

  Suffix "RC" stands for release candidates, suffix "UPDATE" stands for updates
that are still compatible with certain releases.


How to Build x86\_64 Package for Inference (host emulation)
-----------------------------------------------------------
```console
# download and install corresponding PSDK_RTOS to <PSDKR_PATH>
git clone <this_repo>; cd neo-ai-dlr
git checkout <corresponding_tag>
git submodule update --init --recursive

mkdir build_x86; cd build_x86
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DDLR_BUILD_TESTS=OFF ..
make clean; make -j$(nproc)

# build python package in $DLR_HOME/python/dist
cd ..; rm -f build; ln -s build_x86 build
cd python; python3 ./setup.py bdist_wheel; ls dist
```


How to Build aarch64 Package for Inference (target execution)
-------------------------------------------------------------
```console
export ARM64_GCC_PATH=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

# download and install corresponding PSDK_RTOS to <PSDKR_PATH>
git clone <this_repo>; cd neo-ai-dlr
git checkout <corresponding_tag>
git submodule update --init --recursive

mkdir build_aarch64; cd build_aarch64
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DDLR_BUILD_TESTS=OFF -DCMAKE_TOOLCHAIN_FILE=../cmake/ti-aarch64-linux-gcc-toolchain.cmake ..
make clean; make -j$(nproc)

# build python package in $DLR_HOME/python/dist
cd ..; rm -f build; ln -s build_aarch64 build
cd python; python3 ./setup.py bdist_wheel; ls dist
```


Release Details
---------------

#### TIDL\_PSDK\_8.4
- Update 3rdparty/tvm to TIDL\_PSDK\_8.4
