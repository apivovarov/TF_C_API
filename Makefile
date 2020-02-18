CFLAGS=-O3 -march=native -std=c11
LD_FLAGS=-Wl,--rpath=/opt/glibc-2.14/lib -Wl,--dynamic-linker=/opt/glibc-2.14/lib/ld-linux-x86-64.so.2 -L/opt/glibc-2.14/lib


run_cvpr17_tf : cvpr17_tf
	LD_LIBRARY_PATH=/opt/libtensorflow-gpu-glibc214/lib:/usr/local/cuda-10.0/lib64:/opt/TensorRT-7.0.0.11/lib:/usr/lib64:/opt/glibc-2.14/lib ./cvpr17_tf

cvpr17_tf : cvpr17_tf.o
	gcc $(CFLAGS) -o cvpr17_tf cvpr17_tf.o -L/opt/libtensorflow-gpu-glibc214/lib -ltensorflow -ltensorflow_framework $(LD_FLAGS)

cvpr17_tf.o : cvpr17_tf.c /opt/libtensorflow-gpu-glibc214/include
	gcc $(CFLAGS) -o cvpr17_tf.o -c -I/opt/libtensorflow-gpu-glibc214/include cvpr17_tf.c

run_mobilenet : mobilenet
	LD_LIBRARY_PATH=/opt/libtensorflow-gpu-glibc214/lib:/usr/local/cuda-10.0/lib64:/usr/lib64:/opt/glibc-2.14/lib ./mobilenet

mobilenet : mobilenet.o
	gcc $(CFLAGS) -o mobilenet mobilenet.o -L/opt/libtensorflow-gpu-glibc214/lib -ltensorflow -ltensorflow_framework $(LD_FLAGS)

mobilenet.o : mobilenet.c /opt/libtensorflow-gpu-glibc214/include
	gcc $(CFLAGS) -o mobilenet.o -c -I/opt/libtensorflow-gpu-glibc214/include mobilenet.c

clean :
	rm -rf *.o cvpr17_tf mobilenet
