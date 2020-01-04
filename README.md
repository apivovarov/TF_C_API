# Intro

Run Tensorflow style transform model using C API.
Tested on AWS EC2 p3.2xlarge box (sm_70, Tesla V100-SXM2-16GB)

# Preparation
Download style transform model frozen model.

Download [libtensorflow-gpu-10-0.tar.gz](https://pivovaa-us-west-1.s3-us-west-1.amazonaws.com/libtensorflow-gpu-10-0.tar.gz) and extract it to `tensorflow` folder

Install `cuda-10-0` and `libcudnn7`

# Build
```
make cvpr17_tf
```

# Run
```
make run_cvpr17_tf
```
