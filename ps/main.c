#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *argv0) {
    printf("Usage: %s [--input PATH] [--n-frames N]\n", argv0);
}

int main(int argc, char **argv) {
    const char *input = "sample.jpg";
    int frames = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--n-frames") == 0 && i + 1 < argc) {
            frames = atoi(argv[++i]);
        }
    }

    printf("SPECTRA-YOLO public scaffold\n");
    printf("input: %s\n", input);
    printf("frames: %d\n", frames);
    printf("Complete implementation files are retained in the author archive.\n");
    return 0;
}
