
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
#include <iostream>
#include <chrono>
#include <fstream>


using namespace cimg_library;
using namespace std;

int* rgbGPU;
int* rgbBlurGPU;

unsigned char* imageData;
double* mask;
 
__global__ void startKernel() {


}
__global__ void seperateRGB(unsigned char* image, int* resultArray,int width, int height) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < width && row < height) {
        int rowOffset = (width * row * 3);

        int r = (int)image[rowOffset + k * 3];
        int g = (int)image[rowOffset + k * 3 + 1];
        int b = (int)image[rowOffset + k * 3 + 2];

        resultArray[(row * width) + k] = r;
        resultArray[(width * height) + (row * width) + k] = g;
        resultArray[((width * height) * 2) + (row * width) + k] = b;
    }
}

__global__ void Blur(int* rgbArray, int* blurImage, double* mask, int width, int height, int maskSize) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (k < width && row < height) {
        int currentHeight = (row * width);
        int offset = width * height;
        blurImage[currentHeight + k] = 0; //red
        blurImage[offset + currentHeight + k] = 0; //blue
        blurImage[offset * 2 + currentHeight + k] = 0; //green
        double r = 0.;
        double g = 0.;
        double b = 0.;
        for (int i = 0; i < maskSize; i++)
        {
            int maskHeight = (i - ((maskSize - 1) / 2) + row) * width;
            maskHeight = max(maskHeight, 0);
            maskHeight = min(maskHeight, width * height);

            for (int j = 0; j < maskSize; j++) {
                int maskWidth = k + j - ((maskSize - 1) / 2);
                maskWidth = max(maskWidth, 0);
                maskWidth = min(maskWidth, width);
                
                r = r + rgbArray[maskHeight + maskWidth]   * mask[i * maskSize + j];
                g = g + rgbArray[offset + maskHeight + maskWidth]   * mask[i * maskSize + j];
                b = b + rgbArray[offset * 2 + maskHeight + maskWidth]   * mask[i * maskSize + j];         
            }
        }
        blurImage[currentHeight + k] = r;
        blurImage[offset + currentHeight + k] = g;
        blurImage[offset * 2 + currentHeight + k] = b;
    }
}

int width;
int height;
unsigned char* pixelDataChar;
int dataSize;
float maxValue;

void readPPM(string fileName) {
    ifstream file;
    string version;
    file.open(fileName, ios::in | ios::binary);

    if (!file) {
        cerr << "file could not be open" << endl;
        exit(EXIT_FAILURE);
    }

    file >> version;

    // Check version equals P6
    if (version.compare("P6") != 0)
    {
        cout << "Invalid image format (must be 'P6')";
        exit(EXIT_FAILURE);
    }

    file >> width >> height >> maxValue;


    dataSize = height * width * 3;
    pixelDataChar = new unsigned char[dataSize];
    file.get();
    file.read((char*)pixelDataChar, dataSize);

    file.close();
}

void test(bool show)
{
    //Mask
    CImg<double> mask5(5, 5);
    mask5(0, 0) = mask5(0, 4) = mask5(4, 0) = mask5(4, 4) = 1.0 / 256.0;
    mask5(0, 1) = mask5(0, 3) = mask5(1, 0) = mask5(1, 4) = mask5(3, 0) = mask5(3, 4) = mask5(4, 1) = mask5(4, 3) = 4.0 / 256.0;
    mask5(0, 2) = mask5(2, 0) = mask5(2, 4) = mask5(4, 2) = 6.0 / 256.0;
    mask5(1, 1) = mask5(1, 3) = mask5(3, 1) = mask5(3, 3) = 16.0 / 256.0;
    mask5(1, 2) = mask5(2, 1) = mask5(2, 3) = mask5(3, 2) = 24.0 / 256.0;
    mask5(2, 2) = 36.0 / 256.0;
    double maskArray[5 * 5];
    double sum = 0.; 

    //Make the mask vector an array 
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            maskArray[i * 5 + j] = mask5(i, j);
        }
    }


    //Read data from PPM
    readPPM("cake.ppm");


    // Don't use the small cake 
    // CImg<unsigned char> image("cake-small.ppm"), blurimage("cake-small.ppm");
    CImg<unsigned char> image("cake-small.ppm"), blurimage("cake-small.ppm"), blurImageGPU("cake-small.ppm");

    // Use the big cake    
    int* arrayRGB;
    int* arrayBlur;

    arrayRGB = new int[dataSize];
    arrayBlur = new int[dataSize];
    for (int i = 0; i < (height * width) * 3; i++) {
        arrayRGB[i] = -1;
        arrayBlur[i] = -1;
    }
    

    int threads = 4;
    dim3 block_dim(threads, threads, 1);
    dim3 grid_dim(width / threads, height / threads + 1, 1);

    startKernel << <1, 1 >> > ();
    cudaDeviceSynchronize();

    std::cout << "Start GPU" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();

    cudaMalloc((void**)&rgbGPU, dataSize * sizeof(int));
    cudaMalloc((void**)&rgbBlurGPU, dataSize * sizeof(int));

    cudaMalloc((void**)&imageData, dataSize * sizeof(unsigned char));
    cudaMalloc((void**)&mask, 5 * 5 * sizeof(double));

    cudaMemcpy(rgbGPU, arrayRGB, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(imageData, pixelDataChar, dataSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, maskArray, 5 * 5* sizeof(double), cudaMemcpyHostToDevice);

    cudaError_t err;

    seperateRGB << <grid_dim, block_dim >> > (imageData, rgbGPU, width, height);
  /*  err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "error: " << cudaGetErrorString(err) << " err code: " << err  << endl;
        err = cudaSuccess;
        getchar();
    }*/
    cudaDeviceSynchronize();

    Blur << <grid_dim, block_dim >> > (rgbGPU, rgbBlurGPU, mask, width, height, 5);
  /*  err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "error: " << cudaGetErrorString(err) << " err code: " << err  << " func 2" << endl;
        err = cudaSuccess;
    }*/
    cudaThreadSynchronize();


    cudaMemcpy(arrayBlur, rgbBlurGPU, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    int offset = width * height;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            blurImageGPU[i * width + j] = arrayBlur[i * width + j];
            blurImageGPU[offset + (i * width) + j] = arrayBlur[offset + i * width + j];
            blurImageGPU[offset * 2 + i * width + j] = arrayBlur[offset * 2 + i * width + j];
        }
    }



    std::chrono::duration<double> elapsed = end - begin;
    std::cout << "Time taken GPU = " << elapsed.count() << " seconds" << endl;




    // Convolve and record the time taken to do the operation
    auto beginCPU = std::chrono::high_resolution_clock::now();
    // Blur the image!
    blurimage.convolve(mask5);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - beginCPU;
    std::cout << "Time taken to convolve = " << elapsedCPU.count() << " seconds" << endl;
    
    // Show the original and the blurred images and compare.
    
    // To display the images as 400 x 300
    
   /* CImgDisplay main_disp(400, 300, "Original image");
    CImgDisplay main_disp2(400, 300, "Blurred image");
    main_disp.render(image);
    main_disp2.render(blurimage);*/
    

    // Display the images in their original size
    if (show) {
        CImgDisplay main_disp3(blurImageGPU, "Blurred image GPU");
        CImgDisplay main_disp(image, "Original image");
        CImgDisplay main_disp2(blurimage, "Blurred image CPU");

        while (1)
        {
            main_disp.wait(); main_disp2.wait(); main_disp3.wait();
        }
    }

    if (!show) {
        fstream file;
        file.open("TestRun.txt", ios::out | ios::app);

        file << elapsed.count() << ";" << elapsedCPU.count() << "\n";
        file.close();
    }


    delete[] arrayRGB;
    delete[] pixelDataChar;
    delete[] arrayBlur;
}


int main() {
    int times = 1;
    //cout << "How many times do you want to run?" << endl;
    //cin >> times;

    if (times == 1) {
        test(true);
    }
    else {
        for (int i = 0; i < times; i++) {
            test(false);
        }
    }
}

