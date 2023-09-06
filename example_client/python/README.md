## Prepare for server side

```
mkdir models/gpt2lm
copy model.onnx to models/gpt2lm
./utils/OnnxReader models/gpt2lm/model.onnx
./grpc_server
```

## Prepare for python client side

```
cd example_client/python
python3 -m grpc_tools.protoc -I ../../utils --python_out=. --grpc_python_out=. ../../utils/inference.proto ../../utils/onnx.proto
python3 gpt2ml_example.py
```