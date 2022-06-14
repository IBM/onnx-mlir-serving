
FROM onnx/aigrpc-base
ENV GPRC_SRC_DIR=/workdir/grpc/
ENV ONNX_MLIR_DIR=/workdir/onnx-mlir/
ENV ONNX_MLIR_BUILD_DIR=/workdir/onnx-mlir/build/
ENV GPRC_INSTALL_DIR=/workdir/grpc/cmake/build/
RUN cd ${ONNX_MLIR_BUILD_DIR}; make install
RUN wget -O googletest.tar.gz https://github.com/google/googletest/archive/release-1.11.0.tar.gz
RUN tar xf googletest.tar.gz;mv googletest-release-1.11.0 googletest;cd googletest;cmake -DBUILD_SHARED_LIBS=ON .;make
COPY . aigrpc-server
RUN cd aigrpc-server;mkdir -p cmake/build;cd cmake/build;cmake -DGRPC_DIR:STRING=${GPRC_SRC_DIR} -DLOADGEN_DIR:STRING=~/code/aiu/inference/loadgen -DONNX_COMPILER_BUILD_DIR:STRING=${ONNX_MLIR_BUILD_DIR} -ONNX_COMPILER_DIR:STRING=${ONNX_MLIR_DIR}  -DCMAKE_PREFIX_PATH=${GPRC_INSTALL_DIR} ../..
RUN cp -r /workdir/onnx-mlir/include/* /usr/include/
RUN cd aigrpc-server/cmake/build;make -j8
RUN cp googletest/lib/* /usr/lib
RUN cd aigrpc-server/cmake/build/tests;make
RUN cd aigrpc-server/tests; for dir in models/*/; do echo $dir; ls $dir/model.onnx|xargs onnx-mlir; done; cp -r models ../cmake/build/; cp ../cmake/build/tests/grpc-test ../cmake/build
RUN cp aigrpc-server/tests/utest.sh aigrpc-server/cmake/build/
ENTRYPOINT ["/bin/bash"]