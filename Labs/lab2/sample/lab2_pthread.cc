#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long total = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* calculate(void* num){
	unsigned long long* num_list = (unsigned long long*)num;
	unsigned long long cur = num_list[0];
	unsigned long long top = num_list[1];
	unsigned long long r = num_list[2];
	unsigned long long pixels = 0;
	unsigned long long r_sqr = r*r;
	for (unsigned long long x = cur; x < top; x++) {
		unsigned long long y = ceil(sqrtl(r_sqr - x*x));
		pixels += y;
		// pixels %= k;
	}
	pthread_mutex_lock(&mutex);
	total += pixels;
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	pthread_t threads[ncpus];
	unsigned long long amount = r / ncpus;
	
	for(int i = 0; i < ncpus; i++){
		unsigned long long* num = (unsigned long long*)malloc(3 * sizeof(unsigned long long));
		num[0] = i * amount;
		num[1] = (i == ncpus-1) ? r : num[0] + amount;  // top
		num[2] = r;

		pthread_create(&threads[i], NULL, calculate, (void*)num);
	}
	for(int i = 0; i < ncpus; i++){
		pthread_join(threads[i], NULL);
	}
	printf("%llu\n", ((total%k) * 4) % k);
}
