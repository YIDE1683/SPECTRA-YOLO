CROSS ?=
CC := $(CROSS)gcc
CFLAGS := -O2 -Wall -Wextra
BIN := spectra_yolo_public_stub

.PHONY: all clean

all: $(BIN)

$(BIN): main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BIN)
