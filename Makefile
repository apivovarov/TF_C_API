CPPFLAGS=-O3 -march=native -std=c++11
CFLAGS=-O3 -march=native -std=c11
LD_FLAGS=


run_cvpr17_tf : cvpr17_tf
	LD_LIBRARY_PATH=libtensorflow/lib: ./cvpr17_tf

cvpr17_tf : cvpr17_tf.o
	gcc $(CFLAGS) -o cvpr17_tf cvpr17_tf.o -Llibtensorflow/lib -ltensorflow -ltensorflow_framework $(LD_FLAGS)

cvpr17_tf.o : cvpr17_tf.c libtensorflow/include
	gcc $(CFLAGS) -o cvpr17_tf.o -c -Ilibtensorflow/include cvpr17_tf.c

run_mobilenet : mobilenet
	LD_LIBRARY_PATH=libtensorflow/lib ./mobilenet

mobilenet : mobilenet.o
	gcc $(CFLAGS) -o mobilenet mobilenet.o -Llibtensorflow/lib -ltensorflow -ltensorflow_framework $(LD_FLAGS)

mobilenet.o : mobilenet.c libtensorflow/include
	gcc $(CFLAGS) -o mobilenet.o -c -Ilibtensorflow/include mobilenet.c

run_config : config
	LD_LIBRARY_PATH=libtensorflow/lib: ./config

config : config.o
	g++ $(CFLAGS) -o config config.o -Llibtensorflow/lib -ltensorflow -ltensorflow_framework $(LD_FLAGS)

config.o : config.cc libtensorflow/include
	g++ -o config.o -c -Itensorflow/bazel-tensorflow/external/com_google_protobuf/src -Itensorflow/bazel-bin -Ilibtensorflow/include config.cc

clean :
	rm -rf *.o cvpr17_tf mobilenet config
