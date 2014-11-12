#include <stdlib.h>
#include <stdio.h>

#define ALPHABET_SIZE 26

int main(int argc, char **argv) {
	FILE *fp = fopen(argv[1], "r");
	int c;
	int hist[ALPHABET_SIZE] = {0};
	
	while((c = fgetc(fp)) != EOF) {
		c = tolower(c) - 97;
		if(c >= 0 && c < 26) {
			printf("letter: %d\n", c);
			hist[c]++;
		}
	}

	for(c = 0; c < ALPHABET_SIZE; c++)  
		printf("%c: %d\n", c + 97, hist[c]);

	fclose(fp);
}
