#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <bits/stdc++.h>
#include <boost/sort/spreadsort/float_sort.hpp>

#define Left 0
#define Right 1
float *new_data_own;

void Sort(int isRight, float *data_own, int data_own_cnt, float *data_recv, int data_recv_cnt)
{
    if (!isRight)
    {
        int need_cnt = 0, i = 0, j = 0;
        bool flag = false;
        while (i < data_own_cnt && j < data_recv_cnt)
        {
            if (data_own[i] < data_recv[j])
            {
                new_data_own[need_cnt++] = data_own[i++];
            }
            else
            {
                new_data_own[need_cnt++] = data_recv[j++];
            }
            if (need_cnt == data_own_cnt)
            {
                flag = true;
                break;
            }
        }
        if (!flag)
        {
            while (i < data_own_cnt)
            {
                new_data_own[need_cnt++] = data_own[i++];
                if (need_cnt == data_own_cnt)
                {
                    flag = true;
                    break;
                }
            }
        }
        if (!flag)
        {
            while (j < data_recv_cnt)
            {
                new_data_own[need_cnt++] = data_recv[j++];
                if (need_cnt == data_own_cnt)
                {
                    flag = true;
                    break;
                }
            }
        }
    }
    else
    {
        int need_cnt = data_own_cnt - 1, i = data_own_cnt - 1, j = data_recv_cnt - 1;
        bool flag = false;
        while (i >= 0 && j >= 0)
        {
            if (data_own[i] > data_recv[j])
            {
                new_data_own[need_cnt--] = data_own[i--];
            }
            else
            {
                new_data_own[need_cnt--] = data_recv[j--];
            }
            if (need_cnt == -1)
            {
                flag = true;
                break;
            }
        }
        if (!flag)
        {
            while (i >= 0)
            {
                new_data_own[need_cnt--] = data_own[i--];
                if (need_cnt == -1)
                {
                    flag = true;
                    break;
                }
            }
        }
        if (!flag)
        {
            while (j >= 0)
            {
                new_data_own[need_cnt--] = data_recv[j--];
                if (need_cnt == -1)
                {
                    flag = true;
                    break;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Group worldGroup, newGroup;
    MPI_Comm newComm = MPI_COMM_WORLD;

    MPI_File input, output;
    int n = atol(argv[1]);
    int data_cnt = (rank < n % size) ? n / size + 1 : n / size;
    float *data = (float *)malloc(data_cnt * sizeof(float));
    bool doSort, totalDoSort = true;
    new_data_own = (float *)malloc((data_cnt + 1) * sizeof(float));
    int offset = 0;
    for (int i = 0; i < rank; i++)
    {
        offset += (i < n % size) ? (n / size + 1) : (n / size);
    }

    if (size > n)
    {
        int ranges[][3] = {{n, size - 1, 1}};
        size = n;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
        MPI_Group_range_excl(worldGroup, 1, ranges, &newGroup);
        MPI_Comm_create(MPI_COMM_WORLD, newGroup, &newComm);
    }
    if (newComm == MPI_COMM_NULL)
    {
        MPI_Finalize();
        exit(0);
    }

    MPI_File_open(newComm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input);
    if (n / size < 2000)
        MPI_File_read_at(input, offset * sizeof(float), data, data_cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    else
        MPI_File_read_at_all(input, offset * sizeof(float), data, data_cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input);

    // sort
    boost::sort::spreadsort::detail::float_sort(data, &data[data_cnt]);
    while (totalDoSort)
    {
        doSort = false;
        float tmp;
        int data_recv_cnt;
        // even phase
        if (rank % 2 == 0)
        { // even node (left)
            if (rank != size - 1)
            { //////
                MPI_Sendrecv(&data[data_cnt - 1], 1, MPI_FLOAT, rank + 1, 0, &tmp, 1, MPI_FLOAT, rank + 1, 0, newComm, MPI_STATUS_IGNORE);
                if (data[data_cnt - 1] > tmp)
                {
                    data_recv_cnt = (rank + 1 < n % size) ? n / size + 1 : n / size;
                    float *data_recv = (float *)malloc(data_recv_cnt * sizeof(float));
                    MPI_Sendrecv(data, data_cnt, MPI_FLOAT, rank + 1, 0, data_recv, data_recv_cnt, MPI_FLOAT, rank + 1, 0, newComm, MPI_STATUS_IGNORE);
                    Sort(Left, data, data_cnt, data_recv, data_recv_cnt);
                    for (int i = 0; i < data_cnt; i++)
                    {
                        data[i] = new_data_own[i];
                    }
                    doSort = true;
                    free(data_recv);
                }
            }
        }
        else
        { // odd node (right)
            MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank - 1, 0, &tmp, 1, MPI_FLOAT, rank - 1, 0, newComm, MPI_STATUS_IGNORE);
            if (data[0] < tmp)
            {
                doSort = true;
                data_recv_cnt = (rank - 1 < n % size) ? n / size + 1 : n / size;
                float *data_recv = (float *)malloc(data_recv_cnt * sizeof(float));
                MPI_Sendrecv(data, data_cnt, MPI_FLOAT, rank - 1, 0, data_recv, data_recv_cnt, MPI_FLOAT, rank - 1, 0, newComm, MPI_STATUS_IGNORE);
                Sort(Right, data, data_cnt, data_recv, data_recv_cnt);
                for (int i = 0; i < data_cnt; i++)
                {
                    data[i] = new_data_own[i];
                }
                free(data_recv);
            }
        }

        // odd phase
        if (rank % 2 == 1)
        { //odd node (left)
            if (rank != size - 1)
            { ///////
                MPI_Sendrecv(&data[data_cnt - 1], 1, MPI_FLOAT, rank + 1, 0, &tmp, 1, MPI_FLOAT, rank + 1, 0, newComm, MPI_STATUS_IGNORE);
                if (data[data_cnt - 1] > tmp)
                {
                    doSort = true;
                    data_recv_cnt = (rank + 1 < n % size) ? n / size + 1 : n / size;
                    float *data_recv = (float *)malloc(data_recv_cnt * sizeof(float));
                    MPI_Sendrecv(data, data_cnt, MPI_FLOAT, rank + 1, 1, data_recv, data_recv_cnt, MPI_FLOAT, rank + 1, 1, newComm, MPI_STATUS_IGNORE);
                    Sort(Left, data, data_cnt, data_recv, data_recv_cnt);
                    for (int i = 0; i < data_cnt; i++)
                    {
                        data[i] = new_data_own[i];
                    }
                    free(data_recv);
                }
            }
        }
        else
        { //even node (right)
            if (rank != 0)
            { //////
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank - 1, 0, &tmp, 1, MPI_FLOAT, rank - 1, 0, newComm, MPI_STATUS_IGNORE);
                if (data[0] < tmp)
                {
                    doSort = true;
                    data_recv_cnt = (rank - 1 < n % size) ? n / size + 1 : n / size;
                    float *data_recv = (float *)malloc(data_recv_cnt * sizeof(float));
                    MPI_Sendrecv(data, data_cnt, MPI_FLOAT, rank - 1, 1, data_recv, data_recv_cnt, MPI_FLOAT, rank - 1, 1, newComm, MPI_STATUS_IGNORE);
                    Sort(Right, data, data_cnt, data_recv, data_recv_cnt);
                    for (int i = 0; i < data_cnt; i++)
                    {
                        data[i] = new_data_own[i];
                    }
                    free(data_recv);
                }
            }
        }
        MPI_Allreduce(&doSort, &totalDoSort, 1, MPI_C_BOOL, MPI_LOR, newComm);
    }

    MPI_File_open(newComm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output);
    if (n / size < 2000)
        MPI_File_write_at(output, offset * sizeof(float), data, data_cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    else
        MPI_File_write_at_all(output, offset * sizeof(float), data, data_cnt, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output);
    free(data);
    MPI_Finalize();
}