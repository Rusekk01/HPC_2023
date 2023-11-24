import cv2
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
import time


def my_harris(img_dir,window_size,k,threshold):

    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    # Проверка на наличие данного изображения
    if img is None:
        print('Invalid image:' + img_dir)
        return None
    else:
        print('Image successfully read...')
        
    height = img.shape[0]   #.shape[0] высота 
    width = img.shape[1]    #.shape[1] ширина
    matrix_R = np.zeros((height,width))
    
    #   Шаг 1 - Вычисление dx dy
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    # dy, dx = np.gradient(gray)

    #   Шаг 2 - Вычисание dx2 dy2 dxy
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy

    offset = int( window_size / 2 )
    #   Шаг 3 - Вычисление элементов матрицы
    print ("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])
            #   Шаг 4 - Задание матрицы H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])

            #   Шаг 5 - Вычисление значения R
            det=np.linalg.det(H)
            tr=np.matrix.trace(H)
            R=det-k*(tr**2)
            matrix_R[y-offset, x-offset]=R

    print(matrix_R.max(), 'max no cuda')

    #   Шаг 6 - Сравнение с пороговым значением
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value=matrix_R[y, x]
            if value>threshold:
                cv2.circle(img,(x,y),1,(0,0,255))
                
    f1 = plt.figure("Manually implemented Harris detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Manually implemented Harris detector")
    plt.xticks([]), plt.yticks([])
    plt.savefig('My_harris_detector-thresh_%s.png'%(threshold), bbox_inches='tight')
    f1.show()


def CUDA_harris(img_dir,window_size,k,threshold):
    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

    # Проверка на наличие данного изображения
    if img is None:
        print('Invalid image:' + img_dir)
        return None
    else:
        print('Image successfully read...')
        
    height = img.shape[0]   #.shape[0] высота 
    width = img.shape[1]    #.shape[1] ширина
    #matrix_R = np.zeros((height,width))
    
    #   Шаг 1 - Вычисление dx dy
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    # dy, dx = np.gradient(gray)

    #   Шаг 2 - Вычисание dx2 dy2 dxy
    dx2=np.square(dx)
    print(dx2.shape, 'shape')
    dy2=np.square(dy)
    dxy=dx*dy

    offset = int( window_size / 2 )
    #   Шаг 3 - Вычисление элементов матрицы
    print ("Finding Corners...")
    dx2_gpu = cp.asarray(dx2.flatten())
    dy2_gpu = cp.asarray(dy2.flatten())
    dxy_gpu = cp.asarray(dxy.flatten())
    matrix_R_gpu = cp.zeros((height*width,))
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_R(const double* dx2, const double* dy2, const double* dxy, double* matrix_R, const int width, const int height, const int offset, const double k) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
                                                               
    if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
        double Sx2 = 0;
        double Sy2 = 0;
        double Sxy = 0;
                          
        for (int j = y - offset; j <= y + offset; j++) {
            for (int i = x - offset; i <= x + offset; i++) {
                Sx2 += dx2[j * width + i];
                Sy2 += dy2[j * width + i];
                Sxy += dxy[j * width + i];
            }
        }
        double H[2][2] = {{Sx2, Sxy}, {Sxy, Sy2}};
        double det = H[0][0]*H[1][1] - H[0][1]*H[1][0];
        double tr = H[0][0] + H[1][1];
        double R = det - k*(tr*tr);
        matrix_R[y * width + x] = R;
    }
    }
    ''', 'compute_R')

    kernel((32, 32), (32, 32), (dx2_gpu, dy2_gpu, dxy_gpu, matrix_R_gpu, width, height, offset, k))

    #print(matrix_R_gpu)

    matrix_R = np.array(matrix_R_gpu.get()).reshape(height, width)

    print(matrix_R.max(), 'max cuda')

    #   Шаг 6 - Сравнение с пороговым значением
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value=matrix_R[y, x]
            if value>threshold:
                cv2.circle(img,(x,y),1,(0,0,255))
                
    f2 = plt.figure("Manually implemented Harris detector using CUDA")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Manually implemented Harris detector using CUDA")
    plt.xticks([]), plt.yticks([])
    plt.savefig('CUDA_harris_detector-thresh_%s.png'%(threshold), bbox_inches='tight')
    f2.show()

image = cv2.imread('cubes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
print(dst.max())
image[dst>0.01*dst.max()] = [0,0,255]
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


t = time.perf_counter()
CUDA_harris("cubes.jpg", 8, 0.04, 0.45)
gpu_time = time.perf_counter() - t
print(gpu_time, 'gpu_time')


t = time.perf_counter()
my_harris("cubes.jpg", 8, 0.04, 0.45)
cpu_time = time.perf_counter() - t
print(cpu_time, 'cpu_time')


input()