#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


const int INF = ((1 << 30) - 1);
const int V = 6010;

int v, e;
static int D[V][V];

void input(char* fileName){
    FILE* file = fopen(fileName, "rb");
    fread(&v, sizeof(int), 1, file);
    fread(&e, sizeof(int), 1, file);
    printf("%d\n", v);

    // INIT
    for(int i = 0; i < v; i++){
        for(int j = 0; j < v; j++){
            if(i == j) D[i][i] = 0;
            else D[i][j] = INF;
        }
    }

    int edge[3];
    for (int i = 0; i < e; ++i) {
        fread(edge, sizeof(int), 3, file);
        D[edge[0]][edge[1]] = edge[2];
    }
    fclose(file);
}

void output(char* fileName){
    FILE* outfile = fopen(fileName, "w");
    for (int i = 0; i < v; ++i) {
        for (int j = 0; j < v; ++j) {
            if (D[i][j] >= INF) D[i][j] = INF;
        }
        fwrite(D[i], sizeof(int), v, outfile);
    }
    fclose(outfile);
}

int main(int argc, char** argv){
    input(argv[1]);

    for(int k = 0; k < v; k++){
#pragma omp parallel for shared(D)
        for(int i = 0; i < v; i++){
            for(int j = 0; j < v; j++){
                if(D[i][j] > D[i][k] + D[k][j]) 
                    D[i][j] = D[i][k] + D[k][j];
            }
        }
    }

    output(argv[2]);
}


