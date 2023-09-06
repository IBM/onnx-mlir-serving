import grpc
 
import inference_pb2_grpc
import inference_pb2
import onnx_pb2

from transformers import GPT2Model, GPT2Tokenizer
import torch
import torch.nn.functional as F

import numpy as np

def post(out):
    logits = out[0,0, -1, :]
    log_probs = F.softmax(torch.tensor(logits), dim=-1)
    _, prev = torch.topk(log_probs, k=1, dim=-1)
    return prev.tolist()


def run_gpt2(host,maxlen):
    text="Tell me about IBM"
    print(text, end="")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens_ = tokenizer.encode(text)
    tokens = np.array(tokenizer.encode(text))
    for i in range(maxlen):

        options =  [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]

        with grpc.insecure_channel(host,options = options) as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            tensor = onnx_pb2.TensorProto(data_type=7, dims=[1,1,len(tokens_)], int64_data=tokens_)
            response = stub.Inference(inference_pb2.InferenceRequest(tensor=[tensor], model_name='gpt2lm'))

        rdata = response.tensor[0].float_data
        
        prev = post(np.reshape(rdata, response.tensor[0].dims))
        print(tokenizer.decode(prev), end="")
        tokens_.append(prev[0])
 

if __name__ == '__main__':
    run_gpt2('10.1.20.99:50051', 15)