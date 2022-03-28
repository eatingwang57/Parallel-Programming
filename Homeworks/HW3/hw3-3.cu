#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <omp.h>

#define B 32

const int INF = ((1 << 30) - 1);
void input(char *inFileName);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);

int v, e;
int padding;
int *D;

int main(int argc, char *argv[])
{
    input(argv[1]);
    printf("%d\n", padding);
    block_FW();
    output(argv[2]);
    return 0;
}

void input(char *fileName)
{
    FILE *file = fopen(fileName, "rb");
    fread(&v, sizeof(int), 1, file);
    fread(&e, sizeof(int), 1, file);
    printf("%d\n", v);
    padding = v + B - (v % B);
    // D = (int*) malloc(sizeof(int)*padding*padding);
    cudaHostAlloc(&D, sizeof(int) * padding * padding, cudaHostAllocMapped | cudaHostAllocPortable);

    // INIT
    for (int i = 0; i < padding; i++)
    {
        for (int j = 0; j < padding; j++)
        {
            if (i == j)
                D[i * padding + i] = 0;
            else
                D[i * padding + j] = INF;
        }
    }

    int edge[3];
    for (int i = 0; i < e; ++i)
    {
        fread(edge, sizeof(int), 3, file);
        D[edge[0] * padding + edge[1]] = edge[2];
    }
    fclose(file);
}

void output(char *fileName)
{
    FILE *outfile = fopen(fileName, "w");
    for (int i = 0; i < v; ++i)
    {
        for (int j = 0; j < v; ++j)
        {
            if (D[i * padding + j] >= INF)
                D[i * padding + j] = INF;
        }
        fwrite(&D[i * padding], sizeof(int), v, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void Phase1(int *Dist, int r, int v)
{
    int i = threadIdx.y + r * B;
    int j = threadIdx.x + r * B;

    if (i >= v || j >= v)
        return;

    __shared__ int sharedDist[B * B];
    int tIdxY = threadIdx.y * B;
    int idx = tIdxY + threadIdx.x;
    int idxD = i * v + j;
    sharedDist[idx] = Dist[idxD];
    __syncthreads();

    for (int k = 0; k < B; k++)
    {
        int idxY = tIdxY + k;
        int idxX = k * B + threadIdx.x;
        int candidate = sharedDist[idxY] + sharedDist[idxX];
        if (sharedDist[idx] > candidate)
        {
            sharedDist[idx] = candidate;
        }
        __syncthreads();
    }

    Dist[idxD] = sharedDist[idx];
    __syncthreads();
}

__global__ void Phase2(int *Dist, int r, int v)
{
    //  對應到Phase1的那個pivot block
    if (blockIdx.y == r)
        return;

    int i = threadIdx.y + blockIdx.y * B;
    int j = threadIdx.x + blockIdx.y * B;
    int p_i = threadIdx.y + r * B;
    int p_j = threadIdx.x + r * B;

    // blockIdx.x = 0 for row, = 1 for column
    if (blockIdx.x == 0)
    {
        i = p_i;
    }
    else
    {
        j = p_j;
    }

    if (i >= v || j >= v)
        return;

    __shared__ int sharedPivot[B * B];
    __shared__ int sharedSelf[B * B];
    int tIdxY = threadIdx.y * B;
    int idx = tIdxY + threadIdx.x;
    int idxD = i * v + j;
    int idxPivot = p_i * v + p_j;
    sharedPivot[idx] = Dist[idxPivot];
    sharedSelf[idx] = Dist[idxD];
    __syncthreads();

    if (blockIdx.x == 0)
    {
        for (int k = 0; k < B; k++)
        {
            int idxY = tIdxY + k;
            int idxX = k * B + threadIdx.x;
            int candidate = sharedPivot[idxY] + sharedSelf[idxX];
            if (sharedSelf[idx] > candidate)
            {
                sharedSelf[idx] = candidate;
            }
        }
    }
    else
    {
        for (int k = 0; k < B; k++)
        {
            int idxY = tIdxY + k;
            int idxX = k * B + threadIdx.x;
            int candidate = sharedSelf[idxY] + sharedPivot[idxX];
            if (sharedSelf[idx] > candidate)
            {
                sharedSelf[idx] = candidate;
            }
        }
    }

    Dist[idxD] = sharedSelf[idx];
    // __syncthreads();
}
__global__ void Phase3(int *Dist, int r, int v, int offset)
{
    int blkIdx_x = blockIdx.x;
    int blkIdx_y = blockIdx.y + offset;
    if ((blkIdx_x == r) || (blkIdx_y == r))
        return;

    int i = threadIdx.y + blkIdx_y * B;
    int j = threadIdx.x + blkIdx_x * B;

    if (i >= v || j >= v)
        return;

    __shared__ int sharedRow[B * B];
    __shared__ int sharedCol[B * B];
    int idxD = i * v + j;
    int pt = Dist[idxD];
    int tIdxY = threadIdx.y * B;
    int idx = tIdxY + threadIdx.x;
    int idxPivotI = i * v + (threadIdx.x + r * B);
    int idxPivotJ = (threadIdx.y + r * B) * v + j;

    sharedRow[idx] = Dist[idxPivotI];
    sharedCol[idx] = Dist[idxPivotJ];
    __syncthreads();

    for (int k = 0; k < B; k++)
    {
        int idxY = tIdxY + k;
        int idxX = k * B + threadIdx.x;
        int candidate = sharedRow[idxY] + sharedCol[idxX];
        if (pt > candidate)
        {
            pt = candidate;
        }
    }
    Dist[idxD] = pt;
}

void block_FW()
{
    int device_num = 0;
    cudaGetDeviceCount(&device_num);
    int round = ceil(padding, B);
    int *device_D[device_num];
    size_t size = sizeof(int) * padding * padding;
    cudaHostRegister(D, size, cudaHostRegisterDefault);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int round_split = (round % 2 == 1 && thread_id == 1) ? round / 2 + 1 : round / 2;
        int offset = thread_id * round / 2;
        int curPos = offset * B * padding;

        int other_thread = !thread_id;

        cudaSetDevice(thread_id);
        cudaMalloc(&device_D[thread_id], size);
        cudaMemcpy(device_D[thread_id], D, size, cudaMemcpyHostToDevice);

        dim3 block_num1(1, 1);
        dim3 block_num2(2, round);
        dim3 block_num3(round, round_split);
        dim3 thread_num(B, B);

        for (int r = 0; r < round; r++)
        {
            int curRow = r * B * padding;
            if ((r >= offset) && (r < offset + round_split))
            {
                cudaMemcpy(device_D[other_thread] + curRow, device_D[thread_id] + curRow, sizeof(int) * B * padding, cudaMemcpyDeviceToDevice);
            }
#pragma omp barrier

            Phase1<<<block_num1, thread_num>>>(device_D[thread_id], r, padding);
            Phase2<<<block_num2, thread_num>>>(device_D[thread_id], r, padding);
            Phase3<<<block_num3, thread_num>>>(device_D[thread_id], r, padding, offset);
        }
#pragma omp barrier
        cudaMemcpy(D + curPos, device_D[thread_id] + curPos, sizeof(int) * B * round_split * padding, cudaMemcpyDeviceToHost);
    }
}