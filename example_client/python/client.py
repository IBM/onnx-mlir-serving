import grpc
 
import inference_pb2_grpc
import inference_pb2
import onnx_pb2


# import onnxruntime
# import onnx
import numpy as np
# import time
from matplotlib import pyplot as plt

def get_image(image):
    with open(image ,'rb') as f:
        num = np.frombuffer(f.read(), np.float32)
        input_data = num.reshape((1,28,28))
        plt.imshow(input_data[0], interpolation='nearest')
        plt.show()
        print('tt')
    
    return input_data.flatten().tolist()

def run():
    data = get_image('example_client/python/img0.data')
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        tensor = onnx_pb2.TensorProto(data_type=1, dims=[1,1,28,28], float_data=data)
        response = stub.Inference(inference_pb2.InferenceRequest(tensor=[tensor], model_name='mnist'))

    rdata = response.tensor[0].float_data
    r = np.argmax(rdata)
    print("Client received: " + r)
 
 
if __name__ == '__main__':
    run()