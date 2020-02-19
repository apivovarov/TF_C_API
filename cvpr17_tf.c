#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tensorflow/c/c_api.h>

double ms();

TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {                                             
        free(data);                                                                       
}

int load_frozen_model(const char* pb_file, TF_Graph* graph, TF_Status* status);

void print_operations(TF_Graph* graph);

void int_array_to_str(int64_t* arr, int len, char* buff);

int64_t desc_tensor(TF_Tensor* t);

int main() {
  int errC = 0;
  const char* pb_file = "cvpr17_tf114_o.pb";
  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  if ((errC = load_frozen_model(pb_file, graph, status)) != 0) {
    return errC;
  }
  //print_operations(graph);

  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph, opts, status);
  if ((errC = TF_GetCode(status)) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to create Session %s", TF_Message(status));
    return errC;
  }
  printf("Successfully created Session\n");
  TF_DeleteSessionOptions(opts);


  TF_Operation* inputOp = TF_GraphOperationByName(graph, "import/input");
  if (inputOp == NULL) {
    fprintf(stderr, "ERROR: inputOp TF_GraphOperationByName failed %s", TF_Message(status));
    return 1;
  }
 
  TF_Operation* outputOp = TF_GraphOperationByName(graph, "import/Tanh"); 
  if (outputOp == NULL) {
    fprintf(stderr, "ERROR: outputOp TF_GraphOperationByName failed %s", TF_Message(status));
    return 1;
  }

  TF_Output input_0;
  input_0.oper = inputOp;
  input_0.index = 0;
  TF_Output inputs[1] = {input_0};

  TF_Output output_0;
  output_0.oper = outputOp;
  output_0.index = 0;
  TF_Output outputs[1] = {output_0};

  const int H = 480;
  const int W = 640;
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

  TF_Tensor* output_tensors[1]; 
  
  const int N = 100;
  double totalT = 0;
  int64_t osize0 = 0;
  for (int i = -1; i < N; i++) {
    printf("TF_SessionRun...\n");
    double t1 = ms();
    TF_SessionRun(sess,
                  NULL, // Run options.
                  inputs, input_tensors, 1, // Input tensors, input tensor values, number of inputs.
                  outputs, output_tensors, 1, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
    );
    double t2 = ms();

    if ((errC = TF_GetCode(status)) != TF_OK) {
      fprintf(stderr, "ERROR: Session run error: %s", TF_Message(status));
      return errC;
    }

    if (i >= 0) {
        totalT += (t2-t1);
    }
    printf("TF_SessionRun done, Time: %2.3f ms\n", t2-t1);
    if (output_tensors[0] != NULL) {
      TF_Tensor* ot0 = output_tensors[0];
      float* od0 = (float*) TF_TensorData(ot0);
      if (osize0 == 0) {
        osize0 = desc_tensor(ot0);
      }
      printf("Output: %2.3f, %2.3f ... %2.3f, %2.3f\n", od0[0], od0[1], od0[osize0-2], od0[osize0-1]);
    }
  }
  printf("Avg Time: %2.3f ms\n", totalT/N);

  // cleanup
  TF_DeleteTensor(input_tensors[0]);
  TF_DeleteTensor(output_tensors[0]);
  TF_CloseSession(sess, status);
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(sess, status);
  TF_DeleteGraph(graph);
  printf("Cleanup done\n");
  return 0;
}


int load_frozen_model(const char* pb_file, TF_Graph* graph, TF_Status* status) {
  const char* op_prefix = "import";
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


void print_operations(TF_Graph* graph) {
  // Iterate through the operations of a graph
  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) {
    const char* name = TF_OperationName(oper);
    printf("%s\n", name);
  }
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


double ms() {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (tv.tv_sec*1000.0)+(tv.tv_usec/1000.0);
}


void int_array_to_str(int64_t* arr, int len, char* buff) {
  int p = 0;
  p += sprintf(&buff[p], "%s", "[");
  for (int i = 0; i < len; i++) {
      p += sprintf(&buff[p], "%ld,", arr[i]);
  }
  sprintf(&buff[p-1], "%s", "]");
}

int64_t desc_tensor(TF_Tensor* t) {
  int ndim = TF_NumDims(t);
  int64_t shape[ndim];
  int64_t size = 1;
  char shape_str[ndim*2+2];
  for (int d=0; d<ndim; d++) {
    shape[d] = TF_Dim(t, d);
    size *= shape[d];
  }
  int_array_to_str(shape, ndim, shape_str);
  size_t nbytes = TF_TensorByteSize(t);
  printf("NumDims: %d, Shape: %s, Size: %ld, Bytes: %ld\n", ndim, shape_str, size, nbytes);
  return size;
}
