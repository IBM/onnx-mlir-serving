FROM ubuntu:22.04

COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/bin/ /usr/local/bin/
COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/lib/ /usr/local/lib/
COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/include/ /usr/local/include/

RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config cmake git maven
RUN git clone -b v1.46.3 https://github.com/grpc/grpc
RUN cd grpc;git submodule update --init
RUN cd grpc;mkdir -p cmake/build;cd cmake/build;cmake ../..;
RUN cd grpc/cmake/build; make -j8;make install
RUN  rm -rf grpc


RUN apt-get install wget
RUN wget -O googletest.tar.gz https://github.com/google/googletest/archive/release-1.11.0.tar.gz
RUN tar xf googletest.tar.gz;mv googletest-release-1.11.0 googletest;cd googletest;cmake -DBUILD_SHARED_LIBS=ON .;make;make install
RUN rm -rf googletest
RUN ldconfig

ARG WORK_DIR=/workdir
WORKDIR ${WORK_DIR}

COPY . aigrpc-server
RUN cd aigrpc-server;mkdir -p cmake/build;cd cmake/build;cmake -DCMAKE_BUILD_TYPE=Release ../..
RUN cd aigrpc-server/cmake/build;make -j8
RUN cd aigrpc-server/cmake/build/tests;make
RUN cd aigrpc-server/tests; for dir in models/*/; do echo $dir; ls ${dir}model.onnx|xargs onnx-mlir; done; cp -r models ../cmake/build/; cp ../cmake/build/tests/grpc-test ../cmake/build
RUN cd aigrpc-server/java;mvn verify

RUN rm -rf /root/.cache

ENTRYPOINT ["/bin/bash"]