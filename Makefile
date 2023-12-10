CLANG = clang -target ve-linux
CLFLAGS = -O3 -ffast-math -mvepacked
NCC = /opt/nec/ve/bin/ncc

all: sgemv_packed_bf16_unr.o sgemv_bf16_cmo.o sgemv_bf16_cmo_n.o

sgemv_packed_bf16.o: sgemv_packed_bf16.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_packed_bf16_unr.o: sgemv_packed_bf16_unr.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16_cmo.o: sgemv_bf16_cmo.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16_cmo_n.o: sgemv_bf16_cmo_n.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16.o: sgemv_bf16.c
	$(CLANG) $(CLFLAGS) -o $@ -c $<

sgemv_bf16_ve3.o: sgemv_bf16_ve3.c
	$(NCC) -O3 -march=ve3 -mfp16-format=bfloat -fopenmp -mvector-packed -o $@ -c $<

sgemv_bf16_ve3_cmo.o: sgemv_bf16_ve3_cmo.c
	$(NCC) -O3 -march=ve3 -mfp16-format=bfloat -mvector-packed -o $@ -c $<

clean:
	rm -f *.o
