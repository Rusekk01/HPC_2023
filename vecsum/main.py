import cupy as cp
import numpy as np
import time

def vector_sum(a):
    sum = 0
    for i in range(len(a)):
        sum += a[i]
    return sum

list = [10, 1000, 5000, 20000, 100000, 400000, 700000, 1000000]

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vec_sum(const int* a, int* b, const int size) {
    int gridSize = blockDim.x * gridDim.x;
    int first_index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = first_index; index < size; index += gridSize)
    {
        atomicAdd(&b[0], a[index]);
    }
}
''', 'vec_sum')

for size in list:
    print(size, 'size')
    a = cp.ones(size, dtype=int)
    c = cp.zeros(1, dtype=int)
    a_cpu = np.random.random(size)
    t = time.perf_counter()
    add_kernel((1024,), (512,), (a, c, size))
    print(c, 'summa')
    gpu_time = time.perf_counter() - t
    print(gpu_time, 'gpu_time')
    t = time.perf_counter()
    cpu_vec_sum = np.sum(a_cpu)
    cpu_np_time = time.perf_counter() - t
    print(cpu_np_time, ' cpu_np_time')
    t = time.perf_counter()
    vector_sum(a_cpu)
    cpu_time = time.perf_counter() - t
    print(cpu_time, 'cpu_time')