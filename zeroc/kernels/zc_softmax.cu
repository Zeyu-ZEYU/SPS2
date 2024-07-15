#include <torch/extension.h>

__global__ void softmax_kernel(float *input, float *output, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= rows)
        return;

    float max_val = -FLT_MAX;
    int row_start = idx * cols;

    // Step 1: Find the max value in the row to avoid numerical instability
    for (int j = 0; j < cols; ++j)
    {
        if (input[row_start + j] > max_val)
        {
            max_val = input[row_start + j];
        }
    }

    // Step 2: Compute the sum of the exponentials
    float sum_exp = 0;
    for (int j = 0; j < cols; ++j)
    {
        sum_exp += exp(input[row_start + j] - max_val);
    }

    // Step 3: Compute the softmax output
    for (int j = 0; j < cols; ++j)
    {
        output[row_start + j] = exp(input[row_start + j] - max_val) / sum_exp;
    }
}

torch::Tensor softmax(torch::Tensor input)
{
    // Assume input is a 2D float tensor
    int rows = input.size(0);
    int cols = input.size(1);

    torch::Tensor output = torch::empty_like(input);

    // Define CUDA kernel configuration
    int threads_per_block = 256;
    int number_of_blocks = (rows + threads_per_block - 1) / threads_per_block;

    softmax_kernel<<<number_of_blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, py_module)
{
    py_module.def("call", &softmax, "Custom softmax.");
}
