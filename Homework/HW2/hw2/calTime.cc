#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>
#include <time.h>
#include <chrono>
#include <iostream>

int *image;
std::chrono::duration<double> totaltime(0);

typedef enum
{
    immediate_shutdown = 1,
    graceful_shutdown = 2
} threadpool_shutdown_t;

// Struct od the task in the pool
typedef struct
{
    void (*function)(void *);
    void *argument;
} threadpool_task_t;

// Structure of the thread pool
struct threadpool_t
{
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t *threads;
    threadpool_task_t *queue;
    int thread_cnt;
    int count;
    int head;
    int tail;
    int shutdown;
    int started;
};

// Add tasks into the thread pool
int threadpool_add(threadpool_t *pool, void (*function)(void *), void *argument)
{

    if (pool == NULL || function == NULL || pool->shutdown)
    {
        return -1;
    }

    pthread_mutex_lock(&(pool->lock));

    // add task into queue
    pool->queue[pool->tail].function = function;
    pool->queue[pool->tail].argument = argument;
    pool->tail++;
    pool->count += 1;

    pthread_cond_signal(&(pool->notify));
    pthread_mutex_unlock(&pool->lock);

    return 0;
}

// Destroy the thread pool
int threadpool_destroy(threadpool_t *pool, int flag)
{
    if (pool == NULL || pool->shutdown)
    {
        return -1;
    }

    pthread_mutex_lock(&(pool->lock));

    pool->shutdown = (flag) ? graceful_shutdown : immediate_shutdown;

    pthread_cond_broadcast(&(pool->notify));
    pthread_mutex_unlock(&(pool->lock));

    // Join all thread
    for (int i = 0; i < pool->thread_cnt; i++)
    {
        pthread_join(pool->threads[i], NULL);
    }

    if (pool == NULL || pool->started > 0)
    {
        return -1;
    }

    // free the memory in the thread pool
    if (pool->threads)
    {
        free(pool->threads);
        free(pool->queue);

        pthread_mutex_destroy(&(pool->lock));
        pthread_cond_destroy(&(pool->notify));
    }
    free(pool);
    return 0;
}

// Things that a thread does (Grab a task and execute)
void *threadpool_thread(void *Pool)
{
    threadpool_t *pool = (threadpool_t *)Pool;
    threadpool_task_t task;
    std::chrono::duration<double> total(0);
    for (;;)
    {
        pthread_mutex_lock(&(pool->lock));

        // no task waiting in the queue now
        while ((pool->count == 0) && !(pool->shutdown))
        {
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if ((pool->shutdown == immediate_shutdown) ||
            ((pool->shutdown == graceful_shutdown) && (pool->count == 0)))
        {
            break;
        }

        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;

        pool->head++;
        pool->count--;

        pthread_mutex_unlock(&(pool->lock));
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        (*(task.function))(task.argument);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        total = total + time_span;
    }

    pool->started--;

    std::cout << "thread time: " << total.count() << "\n";
    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);

    return (NULL);
}

// Initiate a thread pool
threadpool_t *threadpool_init(int thread_cnt, int queue_size)
{
    threadpool_t *pool = (threadpool_t *)malloc(sizeof(threadpool_t));
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_cnt);
    pool->queue = (threadpool_task_t *)malloc(sizeof(threadpool_task_t) * queue_size);
    pool->head = pool->tail = pool->thread_cnt = pool->count = 0;
    pool->shutdown = pool->started = 0;

    pthread_mutex_init(&(pool->lock), NULL);
    pthread_cond_init(&(pool->notify), NULL);

    for (int i = 0; i < thread_cnt; i++)
    {
        pthread_create(&(pool->threads[i]), NULL, threadpool_thread, (void *)pool);
        pool->thread_cnt++;
        pool->started++;
    }
    return pool;
}

// Calculate Mandelbrot set
void calculate(void *argument)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    double *arguments = (double *)argument;
    int iters = (int)arguments[0];
    double left = arguments[1];
    double right = arguments[2];
    int range = (int)arguments[3];
    double imaginary = arguments[4];
    int pos = (int)arguments[5];

    __m128d x0, y0;
    __m128d x, y, x2, y2;
    __m128d length_squared, int2;
    int pt_cnt = 0;
    int pt_now[2];
    int repeats[2];
    bool fin[2];

    double scale = (right - left) / range;

    for (int k = 0; k < 2; k++)
    {
        x0[k] = pt_cnt * scale + left;
        pt_now[k] = pt_cnt;
        pt_cnt++;
        y0[k] = imaginary;
        x[k] = 0;
        y[k] = 0;
        x2[k] = 0;
        y2[k] = 0;
        repeats[k] = 0;
        length_squared[k] = 0;
        fin[k] = false;
        int2[k] = 2.0;
    }

    while (pt_cnt <= range)
    {
        y = _mm_mul_pd(x, y);
        y = _mm_mul_pd(int2, y);
        y = _mm_add_pd(y, y0);

        x = _mm_sub_pd(x2, y2);
        x = _mm_add_pd(x, x0);

        x2 = _mm_mul_pd(x, x);
        y2 = _mm_mul_pd(y, y);

        length_squared = _mm_add_pd(x2, y2);

        ++repeats[0];
        ++repeats[1];
        if ((repeats[0] >= iters || length_squared[0] >= 4) && !fin[0])
        {
            image[pos * range + pt_now[0]] = repeats[0];
            x[0] = 0;
            y[0] = 0;
            length_squared[0] = 0;
            x2[0] = 0;
            y2[0] = 0;
            repeats[0] = 0;
            x0[0] = pt_cnt * scale + left;
            pt_now[0] = pt_cnt;
            if (pt_cnt >= range)
                fin[0] = true;
            pt_cnt++;
        }

        if ((repeats[1] >= iters || length_squared[1] >= 4) && !fin[1])
        {
            image[pos * range + pt_now[1]] = repeats[1];
            x[1] = 0;
            y[1] = 0;
            length_squared[1] = 0;
            x2[1] = 0;
            y2[1] = 0;
            repeats[1] = 0;
            x0[1] = pt_cnt * scale + left;
            pt_now[1] = pt_cnt;
            if (pt_cnt >= range)
                fin[1] = true;
            pt_cnt++;
        }
    }
    if (!fin[0])
    {
        while (repeats[0] < iters && length_squared[0] < 4)
        {
            y = _mm_mul_pd(x, y);
            y = _mm_mul_pd(int2, y);
            y = _mm_add_pd(y, y0);

            x = _mm_sub_pd(x2, y2);
            x = _mm_add_pd(x, x0);

            x2 = _mm_mul_pd(x, x);
            y2 = _mm_mul_pd(y, y);

            length_squared = _mm_add_pd(x2, y2);
            ++repeats[0];
        }
        image[pos * range + pt_now[0]] = repeats[0];
    }
    if (!fin[1])
    {
        while (repeats[1] < iters && length_squared[1] < 4)
        {
            y = _mm_mul_pd(x, y);
            y = _mm_mul_pd(int2, y);
            y = _mm_add_pd(y, y0);

            x = _mm_sub_pd(x2, y2);
            x = _mm_add_pd(x, x0);

            x2 = _mm_mul_pd(x, x);
            y2 = _mm_mul_pd(y, y);

            length_squared = _mm_add_pd(x2, y2);
            ++repeats[1];
        }
        image[pos * range + pt_now[1]] = repeats[1];
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    totaltime = totaltime + time_span;
}

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);    // for real part
    double right = strtod(argv[4], 0);   // for real part
    double lower = strtod(argv[5], 0);   // for imaginary part
    double upper = strtod(argv[6], 0);   // for imaginary part
    int width = strtol(argv[7], 0, 10);  // for real part
    int height = strtol(argv[8], 0, 10); // for imaginary part

    image = (int *)malloc(width * height * sizeof(int));

    threadpool_t *threadpool = threadpool_init(ncpus, height);

    for (int j = 0; j < height; ++j)
    { // for imaginary part
        double y0 = j * ((upper - lower) / height) + lower;
        double *arg = (double *)malloc(6 * sizeof(double));
        arg[0] = iters;
        arg[1] = left;
        arg[2] = right;
        arg[3] = width;
        arg[4] = y0;
        arg[5] = j;
        int add_err = threadpool_add(threadpool, calculate, (void *)arg);
        // if(!add_err) printf("add errer!\n");
    }

    // 1 for
    int destroy_err = threadpool_destroy(threadpool, 1);
    // if(!destroy_err) printf("destroy errer!\n");

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    std::cout << "avg time: " << totaltime.count() / ncpus << "\n";
}

// 2b
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int *image = (int *)malloc(width * height * sizeof(int));
    int *finalImage = (int *)malloc(width * height * sizeof(int));
    for (int i = 0; i < width * height; i++)
        image[i] = 0;

    double start, end;
    start = omp_get_wtime();
    double scale_y = (upper - lower) / height;
    double scale_x = (right - left) / width;
#pragma omp parallel for shared(image) schedule(static)
    // an iteration handle a row
    for (int j = rank; j < height; j += size)
    {
        __m128d y0, x0;
        __m128d x, y, x2, y2;
        __m128d length_squared, int2;
        int pt_cnt = 0;
        int pt_now[2];
        int repeats[2];
        bool fin[2];

        //Initialize
        for (int k = 0; k < 2; k++)
        {
            y0[k] = j * scale_y + lower;
            x0[k] = pt_cnt * scale_x + left;
            x[k] = 0;
            y[k] = 0;
            x2[k] = 0;
            y2[k] = 0;
            pt_now[k] = pt_cnt;
            length_squared[k] = 0;
            repeats[k] = 0;
            int2[k] = 2.0;
            fin[k] = false;
            pt_cnt++;
        }

        while (pt_cnt <= width)
        {
            y = _mm_mul_pd(x, y);
            y = _mm_mul_pd(int2, y);
            y = _mm_add_pd(y, y0);

            x = _mm_sub_pd(x2, y2);
            x = _mm_add_pd(x, x0);

            x2 = _mm_mul_pd(x, x);
            y2 = _mm_mul_pd(y, y);

            length_squared = _mm_add_pd(x2, y2);

            ++repeats[0];
            ++repeats[1];

            if ((repeats[0] >= iters || length_squared[0] >= 4) && !fin[0])
            {
                // 0的位置的點算到了
                // printf("done 0\n");
                image[j * width + pt_now[0]] = repeats[0];

                // reset
                x0[0] = pt_cnt * scale_x + left;
                x[0] = 0;
                y[0] = 0;
                x2[0] = 0;
                y2[0] = 0;
                repeats[0] = 0;
                length_squared[0] = 0;
                pt_now[0] = pt_cnt;
                if (pt_cnt >= width)
                {
                    fin[0] = true;
                }
                pt_cnt++;
            }

            if ((repeats[1] >= iters || length_squared[1] >= 4) && !fin[1])
            {
                // 1的位置的點算到了
                image[j * width + pt_now[1]] = repeats[1];

                // reset
                x0[1] = pt_cnt * scale_x + left;
                x[1] = 0;
                y[1] = 0;
                x2[1] = 0;
                y2[1] = 0;
                repeats[1] = 0;
                length_squared[1] = 0;
                pt_now[1] = pt_cnt;
                if (pt_cnt >= width)
                {
                    fin[1] = true;
                }
                pt_cnt++;
            }
        }
        if (!fin[0])
        {
            while (repeats[0] < iters && length_squared[0] < 4)
            {
                y = _mm_mul_pd(x, y);
                y = _mm_mul_pd(int2, y);
                y = _mm_add_pd(y, y0);

                x = _mm_sub_pd(x2, y2);
                x = _mm_add_pd(x, x0);

                x2 = _mm_mul_pd(x, x);
                y2 = _mm_mul_pd(y, y);

                length_squared = _mm_add_pd(x2, y2);
                ++repeats[0];
            }
            image[j * width + pt_now[0]] = repeats[0];
        }
        // (Some) point(s) not yet done
        if (!fin[1])
        {
            while (repeats[1] < iters && length_squared[1] < 4)
            {
                y = _mm_mul_pd(x, y);
                y = _mm_mul_pd(int2, y);
                y = _mm_add_pd(y, y0);

                x = _mm_sub_pd(x2, y2);
                x = _mm_add_pd(x, x0);

                x2 = _mm_mul_pd(x, x);
                y2 = _mm_mul_pd(y, y);

                length_squared = _mm_add_pd(x2, y2);
                ++repeats[1];
            }
            image[j * width + pt_now[1]] = repeats[1];
        }
    }
    end = omp_get_wtime();
    double time = end - start;
    double total;
    printf("time: %lf\n", end - start);
    MPI_Reduce(&time, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(image, finalImage, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank)
    {
        write_png(filename, iters, width, height, finalImage);
        free(image);
        printf("avg time: %lf\n", total / size);
    }

    MPI_Finalize();
}
