#ifndef SPECTRA_YOLO_PUBLIC_ACCELERATOR_STUB_H
#define SPECTRA_YOLO_PUBLIC_ACCELERATOR_STUB_H

#include <stdint.h>

typedef struct {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t output_count;
} spectra_public_config_t;

int spectra_public_accelerator_stub(const spectra_public_config_t *config);

#endif
