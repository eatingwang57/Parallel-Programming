#include <assert.h>
#include <stdio.h>
#include <math.h>
# include <mpi.h>

int main(int argc, char** argv) {
	int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0, total = 0;
	unsigned long long amount = r / size;
	unsigned long long cur = rank * amount;
	unsigned long long top = (rank == size-1) ? r : cur + amount;
	for (unsigned long long x = cur; x < top; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
		
	}
	MPI_Reduce(&pixels , &total , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , 0 , MPI_COMM_WORLD);

	if(!rank) printf("%llu\n", (4 * total) % k);

	MPI_Finalize();
}
