CLANG = clang -target ve-linux
CLFLAGS = -O3 -ffast-math -mvepacked

all: sgemv_packed_bf16_unr.o sgemv_bf16_cmo.o

sgemv_packed_bf16.o: sgemv_packed_bf16.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_packed_bf16_unr.o: sgemv_packed_bf16_unr.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16_cmo.o: sgemv_bf16_cmo.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16.o: sgemv_bf16.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

clean:
	rm -f *.o
