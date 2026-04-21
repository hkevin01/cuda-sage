// Naive matrix multiplication — exercises all five transforms:
//   T1: missing __launch_bounds__
//   T2: pointer params lack __restrict__
//   T3: shared memory dims are multiples of 32 (bank conflicts)
//   T4: inner k-loop has constant tile bound (unroll candidate)
//   T5: no threadIdx%N divergence (clean on this transform)

#define BLOCK_SIZE 16
#define TILE_SIZE  16

__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}
