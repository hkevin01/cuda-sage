// Vector addition kernel — naive baseline for transform + tune tests.
// PERF ISSUES (intentional):
//   - No __launch_bounds__  (T1 should inject it)
//   - Pointer params lack __restrict__  (T2 should add it)
//   - BLOCK_SIZE default=256 may not be optimal  (tuner should evaluate)

#define BLOCK_SIZE 256

__global__ void vecadd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}
