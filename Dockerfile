
FROM onnx/aigrpc-base
ENV GPRC_SRC_DIR=/workdir/grpc/
ENV ONNX_MLIR_DIR=/workdir/onnx-mlir/
ENV ONNX_MLIR_BUILD_DIR=/workdir/onnx-mlir/build/
ENV GPRC_INSTALL_DIR=/workdir/grpc/cmake/build/
RUN cd ${ONNX_MLIR_BUILD_DIR}; make install
COPY . aigrpc-server
RUN cd aigrpc-server;mkdir -p cmake/build;cd cmake/build;cmake -DGRPC_DIR:STRING=${GPRC_SRC_DIR} -DLOADGEN_DIR:STRING=~/code/aiu/inference/loadgen -DONNX_COMPILER_BUILD_DIR:STRING=${ONNX_MLIR_BUILD_DIR} -ONNX_COMPILER_DIR:STRING=${ONNX_MLIR_DIR}  -DCMAKE_PREFIX_PATH=${GPRC_INSTALL_DIR} ../..
RUN cp -r /workdir/onnx-mlir/include/* /usr/include/
RUN cd aigrpc-server/cmake/build;make -j8
ENTRYPOINT ["/bin/bash"]