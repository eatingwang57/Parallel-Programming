#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long amount = r/size;
	unsigned long long pixels = 0, total = 0;
	int chunk = (r < 100)? 1 : r/size/4;
	// int chunk = 1;

#pragma omp parallel for schedule(static, chunk) reduction(+: pixels)
	for (unsigned long long x = rank; x < r; x+=size) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}
	MPI_Reduce(&pixels , &total , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , 0 , MPI_COMM_WORLD);
	if(!rank) printf("%llu\n", ((total % k) * 4) % k);
	
	MPI_Finalize();
}
