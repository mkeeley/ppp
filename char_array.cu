#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cuda.h>
#include <math.h>

#define ALPHABET_SIZE 26
#define CHUNK_SIZE 26 
#define MAX_THREADS 2
#define ASCII_CONST 97
#define DEBUG 0

__global__ void compute_hist(char *dev_text, int *dev_hist, int chunk_size, int size) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ int hist[];
	int i = tid * chunk_size,
	    j = 0,
	    text_end = i + chunk_size,
	    offset = threadIdx.x * ALPHABET_SIZE,
	    block_start = blockIdx.x * ALPHABET_SIZE; 
	char c = 0;

	if((tid * ALPHABET_SIZE) >= size) 
		return;
	for(;i<text_end;i++) {
		if((c = dev_text[i]) == '\0') 
			break;
		c -= ASCII_CONST;
		if(c >= 0 && c < ALPHABET_SIZE) 
			hist[c+offset]++;
	}	

#if DEBUG	
	printf("tid: %d, block: %d, start: %d, end: %d, hist offset: %d\n", tid, blockIdx.x, i, text_end, offset);	
	for(i = offset; i < ALPHABET_SIZE + offset; i++)  
		if(hist[i] != 0)
			printf("%d: %c: %d\n", tid, (i%ALPHABET_SIZE)+ ASCII_CONST, hist[i]); 
#endif
	__syncthreads();

	if(tid%MAX_THREADS == 0) 
		for(i = 0; i < ALPHABET_SIZE; i++) 
			for(j = 0; j < MAX_THREADS; j++)
				dev_hist[block_start + i] += hist[i+(j*ALPHABET_SIZE)];
}

__global__ void sum_hist(int *dev_hist, int blocks) {
	int i, j;
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
	printf("\nsum of characters:\n");
	for(i = 0; i < ALPHABET_SIZE; i++) {
		for(j = 1; j < blocks; j++) 
			dev_hist[i] += dev_hist[i + j * ALPHABET_SIZE];
		printf("%c: %d\n", i + 97, dev_hist[i]);
	}
}

int main(int argc, char **argv) {
	FILE *fp;
	char c;
	char *text, 
	     *dev_text;
	int *dev_hist;
	int BLOCKS = 0, 
	    THREADS = 0, 
	    i = 0,
	    sz = 0;

	if(argc != 2) {
		printf("enter file name as first argument\n");
		return 1;
	}

	fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	printf("length of file: %d\n", sz = ftell(fp)+1);
	fseek(fp, 0, SEEK_SET);
	text = (char *)malloc(sz * sizeof(char));

	while((c = fgetc(fp)) != EOF) 
		text[i++] = tolower(c);
	text[i] = '\0';

	printf("chunk size: %d\n", CHUNK_SIZE);
	printf("total threads: %d\n",THREADS = ceil(sz/CHUNK_SIZE));	

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

	compute_hist<<<BLOCKS, MAX_THREADS,MAX_THREADS * ALPHABET_SIZE * sizeof(int)>>>(dev_text, dev_hist, CHUNK_SIZE, (BLOCKS * MAX_THREADS + THREADS) * ALPHABET_SIZE);
	cudaDeviceSynchronize();

	sum_hist<<<1,1>>>(dev_hist, BLOCKS);
	cudaDeviceSynchronize();
	cudaMemcpy(hist, dev_hist, ALPHABET_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_hist);
	cudaFree(dev_text);
}
