#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/c/c_api.h"
#include <string>

int SetConfig(TF_SessionOptions* opts, tensorflow::ConfigProto& config, TF_Status* status_) {
  std::string output;
  if (!config.SerializeToString(&output)) {
    std::cerr << "Failed to Serialize ConfigProto" << std::endl;
    return -1;
  }
  // Dump bytes to console
  const char* o = output.c_str();
  for (int i = 0; i < output.size(); i++) {
    if (i > 0) {printf(",");}
    printf("0x%x", o[i] & 0xff);
  }
  printf("\n");

  // TF_SetConfig
  TF_SetConfig(opts, output.c_str(), output.size(), status_);
  if (TF_GetCode(status_) != TF_OK) {
    std::cerr << "ERROR: TF_SetConfig failed " << TF_Message(status_) << std::endl;
    return -2;
  }
  return 0;
}

int main() {
  tensorflow::ConfigProto config = {};
  tensorflow::GPUOptions* gpu = config.mutable_gpu_options();
  config.set_intra_op_parallelism_threads(2);
  config.set_inter_op_parallelism_threads(3);
  gpu->set_allow_growth(1);
  gpu->set_per_process_gpu_memory_fraction(0.1);
  
  TF_Status* status_ = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  int errC = SetConfig(opts, config, status_);
  // cleanup
  TF_DeleteSessionOptions(opts);
  TF_DeleteStatus(status_);
  return errC;
}
