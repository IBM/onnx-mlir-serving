FROM ubuntu:22.04

COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/bin/ /usr/local/bin/
COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/lib/ /usr/local/lib/
COPY --from=onnxmlirczar/onnx-mlir:latest /usr/local/include/ /usr/local/include/

RUN apt-get update \
     && apt-get install -y build-essential autoconf libtool pkg-config cmake git maven \
     && git clone -b v1.46.3 https://github.com/grpc/grpc \
     && cd grpc; git submodule update --init \
     && cd grpc;mkdir -p cmake/build;cd cmake/build;cmake ../.. \
     && cd grpc/cmake/build; make -j8;make install \
     && rm -rf grpc


ARG WORK_DIR=/workdir
WORKDIR ${WORK_DIR}

COPY . aigrpc-server
RUN cd aigrpc-server \
     && mkdir -p cmake/build; cd cmake/build  \
     && cmake -DCMAKE_BUILD_TYPE=Release ../.. \
     && make -j8 \
     && rm -rf /root/.cache

ENTRYPOINT ["/bin/bash"]