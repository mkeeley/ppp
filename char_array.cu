#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cuda.h>

#define ALPHABET_SIZE 26
#define CHUNK_SIZE 26 
#define LOCAL_HIST_SIZE 1024
#define MAX_THREADS 2
#define DEBUG 0

__global__ void compute_hist(char *dev_text, int *dev_hist, int chunk_size, int size) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ int hist[];
	int i = tid * chunk_size,
	    j = 0,
	    text_end = i + chunk_size,
	    offset = tid * ALPHABET_SIZE,
	    block_start = blockIdx.x * ALPHABET_SIZE * MAX_THREADS, // relative to histogram position
	    block_total = (size/ALPHABET_SIZE)/MAX_THREADS;
	char c = 0;

	if(tid * ALPHABET_SIZE >= size)
		return;
#if DEBUG
	printf("tid: %d, start: %d, end: %d, hist offset: %d\n", tid, i, text_end, offset);	
#endif
	for(;i<text_end;i++) {
		if((c = dev_text[i]) == '\0') 
			break;
		c -= 97;
		if(c >= 0 && c < 26) 
			hist[c+offset]++;
	}	

#if DEBUG	
	for(i = offset; i < ALPHABET_SIZE + offset; i++)  
		if(hist[i] != 0)
			printf("%d: %c: %d\n", tid, (i%26)+ 97, hist[i]); 
#endif
	__syncthreads();

	if(tid%MAX_THREADS == 0) {

		printf("tid: %d, block_start: %d\n", tid, block_start);
		for(i = block_start; i < block_start + ALPHABET_SIZE; i++)
			for(j = 1; j < MAX_THREADS; j++) 
				hist[i] += hist[i+(j*ALPHABET_SIZE)];
	}

	__syncthreads();

	if(tid == 0){
		printf("total blocks: %d\n", block_total);
		for(i = 0; i < block_total; i++)
			printf("\tblock %d:", i);
		printf("\n");
		for(i = 0; i < ALPHABET_SIZE; i++) {
			printf("%c\t", i + 97);
			for(j = 0; j < block_total; j++)
				printf("%d\t\t", hist[i + (j * ALPHABET_SIZE * MAX_THREADS)]);
			printf("\n");
		}
	}
}

int main(int argc, char **argv) {
	FILE *fp;
	int sz;
	int  i=0;
	char c;
	char *text;
	int hist[ALPHABET_SIZE] = {0};	
	int BLOCKS = 0, THREADS = 0;
	int *dev_hist;
	char *dev_text;

	if(argc != 2) {
		printf("enter file name as first argument\n");
		return 1;
	}

	fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	printf("length of file: %d\n", sz = ftell(fp)+1);
	fseek(fp, 0, SEEK_SET);
	text = (char *)malloc(sz * sizeof(char));


	while((c = fgetc(fp)) != EOF) {
		text[i++] = tolower(c);
	}	
	text[i] = '\0';

	cudaMalloc((void **) &dev_hist, ALPHABET_SIZE * sizeof(int));
	cudaMalloc((void **) &dev_text, sz * sizeof(char));

	cudaMemcpy(dev_text, text, sz * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_hist, hist, ALPHABET_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	THREADS = sz/CHUNK_SIZE + 1;
	
	while(THREADS > 0) {
		THREADS -= MAX_THREADS;
		BLOCKS++;
	}

	printf("blocks: %d\n", BLOCKS);
	printf("threads: %d\n", THREADS+4);
	compute_hist<<<BLOCKS, MAX_THREADS,(BLOCKS * MAX_THREADS + THREADS) * 26 * sizeof(int)>>>(dev_text, dev_hist, CHUNK_SIZE, (BLOCKS * MAX_THREADS + THREADS) * 26);
	cudaDeviceSynchronize();

}
