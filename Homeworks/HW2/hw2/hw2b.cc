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

    double scale_y = (upper - lower) / height;
    double scale_x = (right - left) / width;
#pragma omp parallel for shared(image) schedule(dynamic, 1)
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

            // 0的位置的點算到了
            if ((repeats[0] >= iters || length_squared[0] >= 4) && !fin[0])
            {
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

            // 1的位置的點算到了
            if ((repeats[1] >= iters || length_squared[1] >= 4) && !fin[1])
            {
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
        // 0還沒算完
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
        // 1還沒算完
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

    MPI_Reduce(image, finalImage, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank)
    {
        write_png(filename, iters, width, height, finalImage);
        free(image);
    }

    MPI_Finalize();
}