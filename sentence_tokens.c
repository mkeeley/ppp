#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
	FILE *fp;
	size_t len = 0;
	ssize_t read;
	int i = 0;
	char delim[2] = ". ";
	char *token;
	char *buffer;

	if(argv[1] == NULL) {
		printf("no input file\n");
		return 1;
	}
	fp = fopen(argv[1], "r");
	while((read = getdelim(&buffer, &len, '.', fp)) != -1 && read != 1) {
		printf("sentence %d: ", ++i);
		token = strtok(buffer, delim);
		while(token != NULL) {
			printf("%s ",  token);
			token = strtok(NULL, delim);
		}
		printf("\n");
	}

	fclose(fp);
	return 1;
}
	
