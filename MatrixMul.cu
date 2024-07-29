#include <cstdlib>
#include <cassert>
#include<iostream>
// Matrix Multiplication kernel
// Optimizations:o



__global__ void sharedMemoryMatrixMul(int *a, int *b, int *c, int N) {

	__shared__ int A[1024];
	__shared__ int B[1024];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int dim = blockDim.x;

	// Calculate global row and column positions for this thread
	int row = blockIdx.y * dim + ty;
	int col = blockIdx.x * dim + tx;

	int sum = 0;
    //Uses a sliding window, where each thread fills a specific index in the shared memory to reduce redondant global memory calls
	for (int i = 0; i < (N / dim); i++) {
		A[(ty * dim) + tx] = a[row * N + (i * dim + tx)];
		B[(ty * dim) + tx] = b[(i * dim * N + ty * N) + col];

		__syncthreads();
		for (int j = 0; j < dim; j++) {
			sum += A[(ty * dim) + j] * B[(j * dim) + tx];
		}
		__syncthreads();
	}

    // Write back the result
	c[(row * N) + col] = sum;
}



__global__ void simpleMatrixMul(int *a, int *b, int *c, int N){
    // Calculate the row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if((row < N) && (col < N)){
        // Each thread computes one element
        for(int i = 0; i < N; i++){
            c[row * N + col] += a[row * N + i] * b[i * N + col];
        }
    }
}


// Initialize a matrix with random numbers
void create_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

int main(){
    // Problem size
    int N = 1 << 14;
    size_t bytes = N * N * sizeof(int);

    // Allocate host memory (make sure C is zeroed)
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)calloc(N * N, sizeof(int));
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialized host-side matrices
    create_matrix(h_a, N * N);
    create_matrix(h_b, N * N);

    // Copy the matrices over
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    // Set up the CTA and Grid Dimensions
    int threads = 32;
    int blocks = (N + threads -1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Time and test the speed of simple matrix multiplciation
    cudaEventRecord(start, 0);

    simpleMatrixMul<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Simple Memory Elapsed time: " << milliseconds << " ms" << std::endl;

    cudaEventRecord(start, 0);

    sharedMemoryMatrixMul<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the elapsed time
    std::cout << "Shared Memory Elapsed time: " << milliseconds << " ms" << std::endl;
    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    // Copy data back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    return 0;
}
