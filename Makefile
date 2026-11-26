CC      = gcc
CXX     = g++
NVCC    = nvcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -Iinclude
CUDA_ARCH ?= sm_75
NVCCFLAGS = -O3 -std=c++14 -arch=$(CUDA_ARCH) -Iinclude
LDLIBS  = -lm
CUDA_LDFLAGS ?=
CUDA_LDLIBS ?= -lcudart -ldl -lrt -pthread -lm

# `all` continua portavel em maquinas sem CUDA Toolkit. Use `make cuda` e
# `make test_cuda` explicitamente em uma maquina com nvcc e GPU NVIDIA.
.PHONY: all seq omp cuda test test_cuda clean
all: seq omp

seq: bin/kmeans_seq
omp: bin/kmeans_omp
cuda: bin/kmeans_cuda

bin/kmeans_seq: src/main.c src/common.c src/seq.c include/kmeans.h | bin
	$(CC) $(CFLAGS) src/main.c src/common.c src/seq.c $(LDLIBS) -o $@

bin/kmeans_omp: src/main.c src/common.c src/omp.c include/kmeans.h | bin
	$(CC) $(CFLAGS) -fopenmp src/main.c src/common.c src/omp.c $(LDLIBS) -o $@

bin/test_seq: tests/test_seq.c src/common.c src/seq.c include/kmeans.h | bin
	$(CC) $(CFLAGS) tests/test_seq.c src/common.c src/seq.c $(LDLIBS) -o $@

bin/test_omp: tests/test_omp.c src/common.c src/omp.c include/kmeans.h | bin
	$(CC) $(CFLAGS) -fopenmp tests/test_omp.c src/common.c src/omp.c $(LDLIBS) -o $@

bin/test_common: tests/test_common.c src/common.c include/kmeans.h | bin
	$(CC) $(CFLAGS) tests/test_common.c src/common.c $(LDLIBS) -o $@

build/common.o: src/common.c include/kmeans.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/main.o: src/main.c include/kmeans.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/cuda.o: src/cuda.cu include/kmeans.h | build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/test_cuda.o: tests/test_cuda.c include/kmeans.h | build
	$(CC) $(CFLAGS) -c $< -o $@

bin/kmeans_cuda: build/main.o build/common.o build/cuda.o | bin
	$(CXX) $^ $(CUDA_LDFLAGS) $(CUDA_LDLIBS) -o $@

bin/test_cuda: build/test_cuda.o build/common.o build/cuda.o | bin
	$(CXX) $^ $(CUDA_LDFLAGS) $(CUDA_LDLIBS) -o $@

test: bin/test_common bin/test_seq bin/test_omp
	./bin/test_common
	./bin/test_seq
	./bin/test_omp

test_cuda: bin/test_cuda
	./bin/test_cuda

bin:
	mkdir -p bin

build:
	mkdir -p build

clean:
	rm -rf bin build
