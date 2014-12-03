#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cuda.h>
#include <math.h>

#define ALPHABET_SIZE 26
#define CHUNK_SIZE 64
#define MAX_THREADS 64
#define ASCII_CONST 97
#define DEBUG 0

__global__ void compute_hist(char *dev_text, int *dev_hist, int chunk_size, int size, int max) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ int hist[];
	int i = tid * chunk_size,
	    text_end = i + chunk_size,
	    offset = threadIdx.x * ALPHABET_SIZE,
	    block_start = blockIdx.x * ALPHABET_SIZE; 
	int c = 0;
	if(tid > max)
		return;
	for(;i<text_end;i++) {
		if((c = dev_text[i]) == '\0') 
			break;
		c = (c|(1 << 5)) - ASCII_CONST;
		if(c >= 0)  
			if(c < ALPHABET_SIZE)
				hist[c+offset]++;
	}	

#if DEBUG	
	printf("tid: %d, block: %d, start: %d, end: %d, hist offset: %d\n", tid, blockIdx.x, i, text_end, offset);	
	for(i = offset; i < ALPHABET_SIZE + offset; i++)  
		if(hist[i] != 0)
			printf("%d: %c: %d\n", tid, (i%ALPHABET_SIZE)+ ASCII_CONST, hist[i]); 
#endif
	__syncthreads();
	for(i = 0; i < ALPHABET_SIZE; i++) 
		atomicAdd(&dev_hist[i+block_start], hist[i+offset]);
}

__global__ void sum_hist(int *dev_hist, int blocks) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int i;
#if DEBUG
	// keep number of blocks small to maintain formatting
	if(tid == 0) {
		int j;
		printf("total blocks: %d\n\n", blocks);
		for(i = 0; i < blocks; i++)
			printf("\tblock %d:", i);
		printf("\n");
		for(i = 0; i < ALPHABET_SIZE; i++) {
			printf("%c:\t", i + ASCII_CONST);
			for(j = 0; j < blocks; j++)
				printf("%d\t\t", dev_hist[i + (j * ALPHABET_SIZE)]);
				printf("\n");
		}
	}
#endif
	for(i = 1; i < blocks; i++) 
		dev_hist[tid] += dev_hist[tid + i * ALPHABET_SIZE];
	
}

int main(int argc, char **argv) {
	FILE *fp;
	char *text, 
	     *dev_text;
	int *dev_hist;
	int BLOCKS = 0, 
	    THREADS = 0, 
	    sz = 0,
	    i;
	float time_1, time_2;
	cudaEvent_t start, stop;

	if(argc != 2) {
		printf("enter file name as first argument\n");
		return 1;
	}

	fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	printf("length of file: %d\n", sz = ftell(fp)+1);
	fseek(fp, 0, SEEK_SET);
	text = (char *)malloc(sz * sizeof(char));
	
	fread(text, sz, 1, fp);

	printf("chunk size: %d\n", CHUNK_SIZE);
	printf("total threads: %d\n", THREADS = ceil(sz/CHUNK_SIZE));	
	int max = THREADS;

	while(THREADS > 0) {
		THREADS -= MAX_THREADS;
		BLOCKS++;
	}

	int hist[ALPHABET_SIZE * BLOCKS];

	cudaMalloc((void **) &dev_hist, BLOCKS * ALPHABET_SIZE * sizeof(int));
	cudaMalloc((void **) &dev_text, sz * sizeof(char));

	cudaMemcpy(dev_text, text, sz * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemset(dev_hist, 0, BLOCKS*ALPHABET_SIZE * sizeof(int));

	printf("blocks: %d\n", BLOCKS);
	printf("threads per block: %d\n", MAX_THREADS);
	printf("leftover threads: %d\n", THREADS+MAX_THREADS);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	compute_hist<<<BLOCKS, MAX_THREADS,MAX_THREADS * ALPHABET_SIZE * sizeof(int)>>>(dev_text, dev_hist, CHUNK_SIZE, (BLOCKS * MAX_THREADS + THREADS) * ALPHABET_SIZE, max);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_1, start, stop);

	cudaEventRecord(start, 0);

	sum_hist<<<1,ALPHABET_SIZE>>>(dev_hist, BLOCKS);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_2, start, stop);

	cudaMemcpy(hist, dev_hist, ALPHABET_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nsum of characters:\n");
	for(i = 0; i < ALPHABET_SIZE; i++)
		printf("%c: %d\n", i + 97, hist[i]);
	
	printf("time to make buckets: \t%3.3f ms\n", time_1);
	printf("time to sum hist: \t%3.3f ms\n", time_2);
	printf("total time to run: \t%3.3f ms\n", time_1 + time_2);

	cudaFree(dev_hist);
	cudaFree(dev_text);
}
