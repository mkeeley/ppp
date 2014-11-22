#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <cuda.h>

#define ALPHABET_SIZE 26
#define CHUNK_SIZE 32 
#define LOCAL_HIST_SIZE 1024
#define MAX_THREADS 4

__global__ void compute_hist(char *dev_text, int *dev_hist, int chunk_size, int size) {
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ int hist[];
	int i = tid * chunk_size,
	    end = i + chunk_size,
	    offset = tid * ALPHABET_SIZE;
	char c = 0;

	if(tid * ALPHABET_SIZE >= size)
		return;
	printf("tid: %d, start: %d, end: %d, hist offset: %d\n", tid, i, end, offset);	
	for(;i<end;i++) {
		if((c = dev_text[i]) == '\0') 
			break;
		c -= 97;
		if(c >= 0 && c < 26) {
			hist[c+offset]++;
		}	
	}	
	
	for(i = offset; i < ALPHABET_SIZE + offset; i++)  
		if(hist[i] != 0)
			printf("%d: %c: %d\n", tid, (i%26)+ 97, hist[i]); 

	__syncthreads();

	if(tid == 0) {
		printf("size: %d\n", size);
		int j = 0;
		for(i = 0; i < ALPHABET_SIZE; i++)
			for(j=1; j < size/ALPHABET_SIZE; j++) 
				hist[i] += hist[i+(j*ALPHABET_SIZE)];
		printf("total:\n");
		for(i = 0; i < ALPHABET_SIZE; i++)
			if(hist[i] != 0)
				printf("%d: %c: %d\n", tid, i+ 97, hist[i]); 
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
	if(argc != 2) {
		printf("enter file name as first argument\n");
		return 1;
	}

	fp = fopen(argv[1], "r");
	fseek(fp, 0, SEEK_END);
	printf("length of file: %d\n", sz = ftell(fp)+1);
	fseek(fp, 0, SEEK_SET);
	text = (char *)malloc(sz * sizeof(char));

	int *dev_hist;
	char *dev_text;

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
