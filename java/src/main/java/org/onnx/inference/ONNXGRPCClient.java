

package org.onnx.inference;

import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import inference.*;
import java.util.concurrent.TimeUnit;
/**
 * ONNX GRPC Client
 */
public class ONNXGRPCClient {
  
  private static final Logger logger = Logger.getLogger(ONNXGRPCClient.class.getName());

  private final InferenceServiceGrpc.InferenceServiceBlockingStub blockingStub;

  private Inference.InferenceRequest.Builder builder;

  public ONNXGRPCClient(Channel channel) {

    blockingStub = InferenceServiceGrpc.newBlockingStub(channel);
  }

  public void setupRequestBuilder(String model, List<Long> shapes) {
    builder = Inference.InferenceRequest.newBuilder().setModelName(model);
    builder.addAllShape(shapes);
  }


  /** Say hello to server. */
  public void inference(List<Float> data) {
    Inference.InferenceRequest request = builder.addAllData(data).build();
    Inference.InferenceResponse response;
    try {
      response = blockingStub.inference(request);
    } catch (StatusRuntimeException e) {
      logger.log(Level.WARNING, "Inference request failed: {0}", e.getStatus());
      return;
    }
    logger.info("Inference result: " + response.getDataList());
  }



  static class ImageDataset {
    private Map<String, Integer> imageNames = new LinkedHashMap<>();
    private String model;
    public String getModel() {
      return model;
    }

    private int rank;
    public int getRank() {
      return rank;
    }

    private List<Long> shapes;
    public List<Long> getShapes() {
      return shapes;
    }

    private String datasetPath;
    private List<Float>[] images;

    public List<Float>[] getImages() {
      return images;
    }

    public ImageDataset(String datasetPath) throws IOException {
      this.datasetPath = datasetPath;
      readImageList(datasetPath);
      images = new List[imageNames.size()];
      int index = 0;
      for (String imageName:imageNames.keySet()) {
          images[index] = loadImage(imageName, imageNames.get(imageName));
          index++;
      }
    }

    private List<Float> loadImage(String imageName, Integer integer) throws IOException {
      logger.info("Load image:"+imageName);
      Path imagePath = Paths.get(datasetPath, imageName);
      try {
        byte[] imageBytes = Files.readAllBytes(imagePath);
        List<Float> floatList = new ArrayList<Float>();
        // ByteBuffer buffer = ByteBuffer.wrap(imageBytes);
        // List<Double> floatList = new ArrayList<Double>();
        // while (buffer.hasRemaining()) {
        //     floatList.add(buffer.getDouble());
        // }
        for (int i=0; i<imageBytes.length; i=i+4) {
          int asInt = (imageBytes[i] & 0xFF) 
            | ((imageBytes[i+1] & 0xFF) << 8) 
            | ((imageBytes[i+2] & 0xFF) << 16) 
            | ((imageBytes[i+3] & 0xFF) << 24);
          float asFloat = Float.intBitsToFloat(asInt);
          floatList.add(asFloat);
        }
        for (float f: floatList) {
          logger.warning(String.format("%s",f));
        }
        return floatList;
      } catch (IOException ex) {
        logger.warning("Failed to load image:"+imageName);
        throw ex;
      }

    }

    private void readImageList(String datasetPath) throws IOException {
      Path mapFilePath = Paths.get(datasetPath, "val_map.txt");
      Path configFilePath = Paths.get(datasetPath, "config.txt");
      Charset charset = StandardCharsets.US_ASCII;
      try {
          List<String> lines = Files.readAllLines(mapFilePath, charset);
          for(String line: lines) {
              String[] tokens = line.split("\\s+");
              if (tokens.length >1){
                imageNames.put(tokens[0], Integer.parseInt(tokens[1]));
              }
          }

      } catch (IOException ex) {
          logger.warning("Failed to load input dataset map");
          throw ex;
      }
      try {
        List<String> lines = Files.readAllLines(configFilePath, charset);
        model = lines.get(0);
        rank = Integer.parseInt(lines.get(1));
        shapes = new ArrayList<Long>();
        String[] tokens = lines.get(2).split("\\s+");
        for (int i=0; i<tokens.length;i++) {
          shapes.add(Long.parseLong(tokens[i]));
        }
      } catch (IOException ex) {
        logger.warning("Failed to load input data set config file");
        throw ex;
      }
    }

  }

  /**
   * Greet server. If provided, the first element of {@code args} is the name to use in the
   * greeting. The second argument is the target server.
   */
  public static void main(String[] args) throws Exception {

    String server = "localhost:50051";

    String argStr = "";
    for (String arg:args) {
      argStr+=","+arg;
    }
    logger.info("Arguements:"+argStr);
    if (args.length >=1) {
      String datasetPath = args[0];
      ImageDataset dataset = new ImageDataset(datasetPath);
      ManagedChannel channel = ManagedChannelBuilder.forTarget(server)
        .usePlaintext()
        .build();
      try {
        ONNXGRPCClient client = new ONNXGRPCClient(channel);
        client.setupRequestBuilder(dataset.getModel(), dataset.getShapes());
        for ( List<Float> image : dataset.getImages()) {
          client.inference(image);
        }
       
      } finally {
        // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
        // resources the channel should be shut down when it will no longer be used. If it may be used
        // again leave it running.
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
      }
    } else {
      logger.warning("Pls provide dataset path as parameter");
    }

    // Create a communication channel to the server, known as a Channel. Channels are thread-safe
    // and reusable. It is common to create channels at the beginning of your application and reuse
    // them until the application shuts down.
    
  }
}

