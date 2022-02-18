grpc source code: /aivol/grpc/
grpc build: /aivol/grpc_install

##build:

export MY_INSTALL_DIR=/aivol/grpc_install
export PATH="$MY_INSTALL_DIR/bin:$PATH"
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ..
make -j

##run:

server:
```
./AIU_async_server <wait time ns> <num of thread pool>
```
client:
```
./AIU_async_client <file path> 
```
baching run:
./app <input dir> <1 for grpc call, 0 for local call> <target_qps> <useQueue> <num of thread>


example:
1.for accuracy run only (for UT)
./app /aivol/inputs/ccf1_inputs 1
2.for batching grpc call
./app /aivol/inputs/ccf1_inputs 1 1000 0 1000


