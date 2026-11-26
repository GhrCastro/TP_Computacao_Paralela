#define _POSIX_C_SOURCE 200809L
#include "kmeans.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_row(FILE *file, int row, int invalid_binary, int bad_label) {
    for (int j = 0; j < 10; ++j) fprintf(file, "%d,", row * 10 + j);
    for (int j = 10; j < 54; ++j) {
        int value = (j == 10 && invalid_binary) ? 2 : ((row + j) & 1);
        fprintf(file, "%d,", value);
    }
    fprintf(file, "%d\n", bad_label ? 8 : row + 1);
}

static void temporary_path(char path[]) {
    strcpy(path, "/tmp/kmeans-loader-XXXXXX");
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
}

int main(void) {
    char path[64];
    temporary_path(path);
    FILE *file = fopen(path, "w");
    assert(file);
    write_row(file, 0, 0, 0);
    write_row(file, 1, 0, 0);
    assert(fclose(file) == 0);

    Dataset data = {0};
    assert(load_dataset(path, &data) == 0);
    assert(data.n == 2 && data.d == 54 && data.points);
    for (int j = 0; j < 10; ++j) {
        assert(data.points[j] == 0.0f);
        assert(data.points[54 + j] == 1.0f);
    }
    for (int j = 10; j < 54; ++j) {
        assert(data.points[j] == (float)(j & 1));
        assert(data.points[54 + j] == (float)((1 + j) & 1));
    }
    free_dataset(&data);
    assert(!data.points && data.n == 0 && data.d == 0);
    unlink(path);

    temporary_path(path);
    file = fopen(path, "w");
    assert(file); write_row(file, 0, 1, 0); assert(fclose(file) == 0);
    assert(load_dataset(path, &data) != 0);
    unlink(path);

    temporary_path(path);
    file = fopen(path, "w");
    assert(file); write_row(file, 0, 0, 1); assert(fclose(file) == 0);
    assert(load_dataset(path, &data) != 0);
    unlink(path);

    assert(load_dataset(NULL, &data) != 0);
    assert(load_dataset("/arquivo/inexistente", &data) != 0);
    assert(isnan(calculate_inertia(NULL, NULL, 0)));
    puts("Testes do loader e funcoes comuns OK");
    return 0;
}
