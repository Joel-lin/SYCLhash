#!/bin/bash

# TODO: try adding -Wno-deprecated-declarations

rm -fr joelbuild
mkdir joelbuild

cd joelbuild

PRE=/opt/intel/oneapi/compiler/2024.2

export CXXFLAGS="-fsycl -fsycl-targets=intel_gpu_pvc -I/opt/intel/oneapi/2024.2/include -I/opt/intel/oneapi/2024.2/include/sycl"

#export CXXFLAGS="-I${MPICH_DIR}/include"
#export LDFLAGS="-L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa"

# need release build type becaue -g makes the compile super slow
cmake -DCMAKE_PREFIX_PATH=$PRE \
      -DCMAKE_CXX_COMPILER=$PRE/bin/icpx \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_DOCS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PRE \
      ..

make -j4

