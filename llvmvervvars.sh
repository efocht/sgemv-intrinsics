LLVM_INSTALL=/usr/local/ve/llvm-ve-rv-2.2.0

### configure common paths ###
export CPATH=${LLVM_INSTALL}/include:${CPATH}
export LD_LIBRARY_PATH=${LLVM_INSTALL}/lib:${LD_LIBRARY_PATH}
export LD_RUN_PATH=${LLVM_INSTALL}/lib:${LD_RUN_PATH}
export LIBRARY_PATH=${LLVM_INSTALL}/lib:${LIBRARY_PATH}
export CMAKE_MODULE_PATH=${LLVM_INSTALL}/share
export PATH=${LLVM_INSTALL}/bin:${PATH}

export PKG_CONFIG_PATH=${LLVM_INSTALL}/lib/pkgconfig:${PKG_CONFIG_PATH}
