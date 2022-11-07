#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

//#define CACHE_SIZE 2*1024*1024
#define NS_PER_S (1000000000L)

unsigned long int getNs() {
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec*NS_PER_S + ts.tv_nsec;
}

void remove_all_chars(char* str, char c) {
	char *pr = str, *pw = str;
	while (*pr) {
		*pw = *pr++;
		pw += (*pw != c);
	}
	*pw = '\0';
}

int main(int argc, char **argv) {
	char line[512], buffer[32];
	long long int column;
	FILE *meminfo;

	if (!(meminfo = fopen("/home/lucz/iBench-master/testFile", "r"))) {
		perror("/proc/meminfo: fopen");
		return -1;
	}

	while (fgets(line, sizeof(line), meminfo)) {
		column = atoll(line) / 1000;
		printf("Cgroup Memory Size is: %lld KB\n", column);
	}

	if (!(meminfo = fopen("/proc/meminfo", "r"))) {
		perror("/proc/meminfo: fopen");
		return -1;
	}

	while (fgets(line, sizeof(line), meminfo)) {
		if (strstr(line, "MemTotal")) {
			char* colStr;
			colStr = strstr(line, ":");
			remove_all_chars(colStr, ':'); 
			remove_all_chars(colStr, 'k'); 
			remove_all_chars(colStr, 'B');
			remove_all_chars(colStr, ' ');
			column = atoi(colStr);
		        column = 1000*column;	
			fclose(meminfo);
			printf("the number is: %lld\n", column);
			return column; 
		}
	}
	fclose(meminfo);
	return 0;
}

