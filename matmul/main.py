import cupy as cp
import numpy as np
import time


def cpu_matmul(a, b, n):
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] = 0
            for k in range(n):
                c[i, j] += a[i, j]*b[i, j]
    return c

list = [50, 100, 200, 400, 800, 1200, 1600, 1800, 2000]

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void matmul(const float* a, const float* b, float* c, int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float value = 0.0, elem1 = 0.0, elem2 = 0.0;
    
    for(int i = 0; i < width; i++)
	{
		elem1 = a[y * width + i];
		elem2 = b[i * width + x];
		
		value += elem1 * elem2;
	}
                      
    c[y * width + x] = value;
}
''',
"matmul")


for size in list:
    print(size, 'size')
    a = cp.random.random((size, size))
    b = cp.random.random((size, size))
    c = cp.zeros((size, size))
    a_cpu = np.random.random((size, size))
    b_cpu = np.random.random((size, size))
    t = time.perf_counter()
    res = add_kernel((1000, 1000), (10, 10), (a, b, c, size))
    gpu_time = time.perf_counter() - t
    print(gpu_time, 'gpu_time')
    t = time.perf_counter()
    cpu_matmul_np = np.matmul(a_cpu, b_cpu)
    cpu_np_time = time.perf_counter() - t
    print(cpu_np_time, ' cpu_np_time')
    t = time.perf_counter()
    cpu_matmul(a_cpu, b_cpu, size)
    cpu_time = time.perf_counter() - t
    print(cpu_time, 'cpu_time')
