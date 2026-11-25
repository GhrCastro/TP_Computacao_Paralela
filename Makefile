# Build minimo para seq e OpenMP (alvo cuda fica para a integracao).
CC      = gcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -Iinclude
LDLIBS  = -lm

.PHONY: all seq omp test clean
all: seq omp

seq: bin/kmeans_seq
omp: bin/kmeans_omp

bin/kmeans_seq: src/main.c src/common.c src/seq.c include/kmeans.h | bin
	$(CC) $(CFLAGS) src/main.c src/common.c src/seq.c $(LDLIBS) -o $@

bin/kmeans_omp: src/main.c src/common.c src/omp.c include/kmeans.h | bin
	$(CC) $(CFLAGS) -fopenmp src/main.c src/common.c src/omp.c $(LDLIBS) -o $@

bin/test_seq: tests/test_seq.c src/common.c src/seq.c include/kmeans.h | bin
	$(CC) $(CFLAGS) tests/test_seq.c src/common.c src/seq.c $(LDLIBS) -o $@

bin/test_omp: tests/test_omp.c src/common.c src/omp.c include/kmeans.h | bin
	$(CC) $(CFLAGS) -fopenmp tests/test_omp.c src/common.c src/omp.c $(LDLIBS) -o $@

test: bin/test_seq bin/test_omp
	./bin/test_seq
	./bin/test_omp

bin:
	mkdir -p bin

clean:
	rm -rf bin
