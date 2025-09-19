#include "elementwise_cpp.h"

#define CHECK_TORCH_TENSOR_DTYPE(T, torch_type)                                  \
  if (((T).options().dtype() != (torch_type))) {                                 \
    std::cout << "Tensor Info:" << (T).options().dtype() << std::endl;           \
    throw std::runtime_error("value must be " #torch_type);                      \
  }

#define TORCH_BINDING_ELEM_ADD(packed_type, element_type)                                   \
  __global__ void elementwise_add_##packed_type(element_type *a,                            \
                                                element_type *b,                            \
                                                element_type *c, int N) {                   \
    elementwise_add_##packed_type##_kernel(a, b, c, N);                                     \
  }

TORCH_BINDING_ELEM_ADD(f32, float)
TORCH_BINDING_ELEM_ADD(f32x4, float)
TORCH_BINDING_ELEM_ADD(f16, half)
TORCH_BINDING_ELEM_ADD(f16x2, half)
TORCH_BINDING_ELEM_ADD(f16x8, half)
TORCH_BINDING_ELEM_ADD(f16x8_pack, half)

template<typename ElementType, torch::Dtype TorchType, int n_elements>
void elementwise_add(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     void (*kernel)(ElementType *, ElementType *, ElementType *, int)
) {
    CHECK_TORCH_TENSOR_DTYPE(a, TorchType);
    CHECK_TORCH_TENSOR_DTYPE(b, TorchType);
    CHECK_TORCH_TENSOR_DTYPE(c, TorchType);
    const int ndim = a.dim();
    if (ndim != 2) {
        int N = 1;
        for (int i = 0; i < ndim; i++) {
            N *= a.size(i);
        }
        dim3 block(256 / (n_elements));
        dim3 grid((N + 256 - 1) / 256);
        kernel<<<grid, block>>>(
            reinterpret_cast<ElementType *>(a.data()),
            reinterpret_cast<ElementType *>(b.data()),
            reinterpret_cast<ElementType *>(c.data()),
            N);
    } else {
        const int S = a.size(0);
        const int K = a.size(1);
        const int N = S * K;
        if (K / n_elements <= 1024) {
            dim3 block(K / n_elements);
            dim3 grid(S);
            kernel<<<grid, block>>>(
                reinterpret_cast<ElementType *>(a.data()),
                reinterpret_cast<ElementType *>(b.data()),
                reinterpret_cast<ElementType *>(c.data()),
                N);
        } else {
            int N = 1;
            for (int i = 0; i < ndim; i++) {
                N *= a.size(i);
            }
            dim3 block(256 / (n_elements));
            dim3 grid((N + 256 - 1) / 256);
            kernel<<<grid, block>>>(
                reinterpret_cast<ElementType *>(a.data()),
                reinterpret_cast<ElementType *>(b.data()),
                reinterpret_cast<ElementType *>(c.data()),
                N);
        }
    }
}

int main() {
}
