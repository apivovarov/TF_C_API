CFLAGS=-O3 -march=native -std=c11

run_cvpr17_tf : cvpr17_tf
	LD_LIBRARY_PATH=tensorflow/lib ./cvpr17_tf

cvpr17_tf : cvpr17_tf.o
	gcc $(CFLAGS) -o cvpr17_tf cvpr17_tf.o -Ltensorflow/lib -ltensorflow

cvpr17_tf.o : cvpr17_tf.c tensorflow/include
	gcc $(CFLAGS) -o cvpr17_tf.o -c -Itensorflow/include cvpr17_tf.c

clean_cvpr17_tf :
	rm -rf cvpr17_tf.o cvpr17_tf
