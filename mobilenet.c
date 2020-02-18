#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>

TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {                                             
        free(data);                                                                       
}

int load_frozen_model(const char* pb_file, TF_Graph* graph, TF_Status* status);

int main() {
  int errC = 0;
  const char* pb_file = "mobilenet_v1_1.0_224_frozen.pb";
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  load_frozen_model(pb_file, graph, status);
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, opts, status);
  TF_DeleteSessionOptions(opts);
  TF_Output input_0;
  input_0.oper = TF_GraphOperationByName(graph, "input");
  input_0.index = 0;
  TF_Output inputs[1] = {input_0};
  TF_Output output_0;
  output_0.oper = TF_GraphOperationByName(graph, "MobilenetV1/Predictions/Reshape_1");
  output_0.index = 0;
  TF_Output outputs[1] = {output_0};

  const int H = 224;
  const int W = 224;
  const int len = H * W * 3;
  const int num_bytes = len * sizeof(float);
  int64_t dims[] = {1, H, W, 3};
  TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, dims, 4, num_bytes);
  float* input_data = (float*)TF_TensorData(input_tensor);
  // dummy input data 
  for (int i = 0 ; i < len; i++) {
    input_data[i] = 0.1;
  }
  TF_Tensor* input_tensors[1] = {input_tensor}; 
  TF_Tensor* output_tensors[1] = {}; 
  
  const int N = 20000;
  for (int i = 0; i < N; i++) {
    printf("%d TF_SessionRun...\n", i);
    TF_SessionRun(sess,
                  NULL, // Run options.
                  inputs, input_tensors, 1, // Input tensors, input tensor values, number of inputs.
                  outputs, output_tensors, 1, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
    );
    TF_Tensor* ot0 = output_tensors[0];
    float* od0 = (float*) TF_TensorData(ot0);
    // Output Tensor is new object each time, TensorData is new array each time
    printf("Output Tensor: %p, output Tensor Data: %p\n", ot0, od0);
    // !!! uncomment the line below to stop GPU memory leak
    // Output Tensor will be new object each time, but TensorData will be the same (will be reused) 
    // TF_DeleteTensor(ot0);
  }

  // cleanup
  TF_DeleteTensor(input_tensors[0]);
  TF_DeleteTensor(output_tensors[0]);
  TF_CloseSession(sess, status);
  TF_DeleteSession(sess, status);
  TF_DeleteGraph(graph);
  printf("Cleanup done\n");
  return 0;
}


int load_frozen_model(const char* pb_file, TF_Graph* graph, TF_Status* status) {
  const char* op_prefix = "";
  int errC = 0;
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_ImportGraphDefOptionsSetPrefix(opts, op_prefix);

  TF_Buffer* graph_def = read_file(pb_file);
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  if ((errC = TF_GetCode(status)) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
    return errC;
  }
  printf("Successfully imported graph\n");
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(graph_def);
  return 0;
}

TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0L, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0L, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  if (fread(data, fsize, 1, f) != 1) {
    printf("File read error....\n");
  }
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();                                                        
  buf->data = data;
  buf->length = fsize;                                                                    
  buf->data_deallocator = free_buffer;                                                    
  return buf;
}

