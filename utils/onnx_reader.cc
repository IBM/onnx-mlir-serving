#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "inference.pb.h"
#include "onnx.pb.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>

using google::protobuf::io::FileOutputStream;
using google::protobuf::io::FileInputStream;


void createModelConfig(const char* file_path, const char* out_put_file){
  std::ifstream input(file_path,std::ios::ate | std::ios::binary); // open file and move current position in file to the end
  std::streamsize size = input.tellg(); // get current position in file
  input.seekg(0,std::ios::beg); // move to start of file
  std::vector<char> buffer(size);
  input.read(buffer.data(),size); // read raw data
  onnx::ModelProto model;
  model.ParseFromArray(buffer.data(),size); // parse protobuf
  auto graph = model.graph();

  inference::ModelConfig modelConfig;

  std::vector<std::string> initializer_name;
  for(auto initializer: graph.initializer()){
    initializer_name.emplace_back(initializer.name());
  }

  for(auto input_data: graph.input()){
    bool match = false;
    for(std::string n: initializer_name){
      if(input_data.name().compare(n)==0){
        match = true;
        break;
      }
    }
    if(!match){
      auto input = modelConfig.add_input();
      input->CopyFrom(input_data);
    }

  }

  for(auto output_data: graph.output()){
    bool match = false;
    for(std::string n: initializer_name){
      if(output_data.name().compare(n)==0){
        match = true;
        break;
      }
    }
    if(!match){
      auto output = modelConfig.add_output();
      output->CopyFrom(output_data);
    }
  }

  modelConfig.set_max_batch_size(1);


  int fd = open(out_put_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* outputfile = new FileOutputStream(fd);
  google::protobuf::TextFormat::Print(modelConfig, outputfile);
  outputfile->Flush();
  close(fd);

}


int main(int argc, char** argv) {
    std::string input_path = argv[1];
    int found = input_path.find_last_of("/\\");
    std::string output_path = "config";
    if(found > 0){
      output_path = input_path.substr(0,found) + "/config";
    }
    std::cout << "input path: "  << input_path << '\n';
    std::cout << "output path: " << output_path << '\n';
    createModelConfig(input_path.c_str(), output_path.c_str());

}