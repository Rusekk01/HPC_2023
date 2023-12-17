import cupy as cp
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import time

def my_sp (img_dir, filter_size):
    data = np.array(Image.open(img_dir).convert("L"))
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    data_final = np.pad(data_final, pad_width=indexer, mode='edge')
    data = np.pad(data, pad_width=indexer, mode='edge')
    for i in range(indexer, len(data) - indexer):
        for j in range(indexer, len(data[0]) - indexer):
            for z in range(filter_size):
                for k in range(filter_size):
                    temp.append(data[i + z - indexer][j + k - indexer]) 
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    f1 = plt.figure("Manually implemented Salt and Pepper algorithm")
    plt.imshow(data_final[indexer:-indexer, indexer:-indexer], cmap='gray'), plt.title("Manually implemented Salt and Pepper algorithm")
    plt.xticks([]), plt.yticks([])
    img_final = Image.fromarray(data_final[indexer:-indexer, indexer:-indexer])
    img_final.convert("L").save('My_salt_and_pepper_%s.bmp'%(filter_size))
    #plt.show()

def cuda_sp (img_dir, filter_size):
    data = np.array(Image.open(img_dir).convert("L"))
    temp = []
    indexer = filter_size // 2
    data = np.pad(data, pad_width=indexer, mode='edge')
    data_gpu = cp.asarray(data.flatten())
    result_gpu = cp.zeros(data_gpu.shape[0], dtype=cp.uint8)
    kernel = cp.RawKernel(r'''
    extern "C" 
    __global__ void m_filter(unsigned char* input, unsigned char* output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            unsigned char window[9];
            int index = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        //printf("%u", input[ny * width + nx]);
                        window[index] = input[ny * width + nx];
                    } else {
                        window[index] = 0;
                    }

                    index++;
                }
            }

            for (int i = 0; i < 9; i++) {
                for (int j = i + 1; j < 9; j++) {
                    if (window[i] > window[j]) {
                        unsigned char temp = window[i];
                        window[i] = window[j];
                        window[j] = temp;
                    }
                }
            }
            output[y * width + x] = window[4];
        }
    }
    ''',
    'm_filter')

    kernel((32,32), (32,32), (data_gpu, result_gpu, data.shape[1], data.shape[0]))

    data_final = np.array(result_gpu.get()).reshape(len(data), len(data[0]))

    f1 = plt.figure("CUDA Salt and Pepper algorithm")
    plt.imshow(data_final[indexer:-indexer, indexer:-indexer], cmap='gray'), plt.title("CUDA Salt and Pepper algorithm")
    plt.xticks([]), plt.yticks([])
    img_final = Image.fromarray(data_final[indexer:-indexer, indexer:-indexer])
    img_final.convert("L").save('My_CUDA_salt_and_pepper_%s.bmp'%(filter_size))
    #plt.show()

t = time.perf_counter()
cuda_sp("impulse-noise.bmp", 3)
cpu_time = time.perf_counter() - t
print(cpu_time, 'gpu_time')

t = time.perf_counter()
my_sp("impulse-noise.bmp", 3)
cpu_time = time.perf_counter() - t
print(cpu_time, 'cpu_time')
