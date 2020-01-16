#include <cstdio>
#define ARR_LEN 64

__global__ void reverse_arr(int *d_arr, int *d_out) {
    d_out[ARR_LEN - threadIdx.x - 1] = d_arr[threadIdx.x];
}

void print_arr(int *arr, int size){
    int i = 0;
    while (i < size) {
        std::printf("%d ", arr[i++]);
    }

    std::printf("\n");
}

void fill_arr(int *arr, int len) {
    int i;
    for (i=0; i<len; i++) {
        arr[i] = i;
    }
}

int main(void) {
    std::printf("Hello, world!");

    int *h_arr;
    int *d_arr, *d_out; 
    int size = ARR_LEN * sizeof(int);

    cudaMalloc((void**) &d_arr, size);
    cudaMalloc((void**) &d_out, size);
    h_arr = (int *)malloc(size);

    fill_arr(h_arr, ARR_LEN);

    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    std::printf("Reversing array\n");

    reverse_arr<<<1,ARR_LEN>>>(d_arr, d_out);

    cudaMemcpy(h_arr, d_out, size, cudaMemcpyDeviceToHost);
    print_arr(h_arr, ARR_LEN);

    cudaFree(d_arr); cudaFree(d_out);
    return 0;
}