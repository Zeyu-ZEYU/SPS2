#include <iostream>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace std;

__global__ void device_kernel(at::Half *hh)
{
    half y = __float2half(hh[0]);
    half t = __hmul(y, y);
    hh[0] = __half2float(t);
}

void zeyu_cuda_tests()
{
    at::Half x = 3.1786;

    at::Half y = x * x;
    cout << y << "\n";

    at::Half *d_o;
    at::Half *h_o = new at::Half[1];
    cudaMalloc((void **)&d_o, 1 * sizeof(at::Half));
    h_o[0] = 3.1786;
    cudaMemcpy(d_o, h_o, 1 * sizeof(at::Half), cudaMemcpyHostToDevice);
    device_kernel<<<1, 1>>>(d_o);

    cudaMemcpy(h_o, d_o, 1 * sizeof(at::Half), cudaMemcpyDeviceToHost);
    cout << h_o[0] << "\n";
    cudaFree(d_o);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("call", &zeyu_cuda_tests, "Zeyu CUDA Tests");
}
