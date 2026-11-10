/*
 * Buffer layout and the reverse alpha-over traversal follow the local
 * diff-gaussian-rasterization backend by Inria / GRAPHDECO (2023).
 * See copied backend LICENSE.md. Research/evaluation use only.
 * This separate analyzer never writes any original renderer buffer.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <vector>

namespace {
template <class T>
T* take(char*& ptr, size_t count, const char* end) {
    uintptr_t p=(reinterpret_cast<uintptr_t>(ptr)+127)&~uintptr_t(127);
    T* result=reinterpret_cast<T*>(p);
    ptr=reinterpret_cast<char*>(result+count);
    TORCH_CHECK(ptr<=end,"Renderer state buffer prefix too short");
    return result;
}

__global__ void moments(
    const float2* xy, const float4* conic, const float* colors,
    const uint32_t* point_list, const uint2* ranges,
    const float* final_T, const uint32_t* n_contrib,
    const float* rgb, const float* gt, const float* grad, const float* bg,
    int P, int R, int W, int H, double* output, float* audit, int* invalid) {
    const int x=blockIdx.x*16+threadIdx.x, y=blockIdx.y*16+threadIdx.y;
    if (x>=W || y>=H) return;
    const int pixel=y*W+x, size=W*H;
    const int tiles_x=(W+15)/16;
    const uint2 range=ranges[blockIdx.y*tiles_x+blockIdx.x];
    if (range.x>range.y || range.y>uint32_t(R)) {
        atomicAdd(invalid,1);return;
    }
    const uint32_t last=n_contrib[pixel];
    if (last>range.y-range.x) {atomicAdd(invalid+1,1);return;}
    float T=final_T[pixel];
    float tail[3]={bg[0],bg[1],bg[2]};
    const double norm=1.0/(3.0*double(size));
    for (int64_t at=int64_t(range.x)+int64_t(last)-1; at>=int64_t(range.x);--at) {
        const uint32_t id=point_list[at];
        if (id>=uint32_t(P)) {atomicAdd(invalid+2,1);return;}
        const float2 center=xy[id];const float4 c=conic[id];
        const float dx=center.x-float(x),dy=center.y-float(y);
        const float power=-.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
        if (power>0.f) continue;
        const float raw_alpha=c.w*expf(power);
        const float alpha=fminf(.99f,raw_alpha);
        if (alpha<1.f/255.f) continue;
        T=T/(1.f-alpha);
        double linear=0.,q=0.,perceptual=0.,finite_linear=0.,finite_q=0.;
        for (int ch=0;ch<3;++ch) {
            const float color=colors[id*3+ch];
            const double local=double(T)*double(color-tail[ch]);
            const float base=rgb[ch*size+pixel];
            const float clipped=fminf(1.f,fmaxf(0.f,base));
            // Match the native backward's straight-through opacity cap.
            // Match torch.clamp's local derivative for RGB risk as well.
            const double direction=(base>=0.f && base<=1.f)?double(raw_alpha)*local:0.;
            const double finite_direction=double(alpha)*local;
            const double error=double(clipped)-double(gt[ch*size+pixel]);
            linear += -2.*error*direction*norm;
            q += direction*direction*norm;
            perceptual += -double(grad[ch*size+pixel])*direction;
            finite_linear += -2.*error*finite_direction*norm;
            finite_q += finite_direction*finite_direction*norm;
            tail[ch]=alpha*color+(1.f-alpha)*tail[ch];
        }
        double* row=output+int64_t(id)*8;
        atomicAdd(row+0,linear);atomicAdd(row+1,q);atomicAdd(row+2,perceptual);
        atomicAdd(row+3,double(alpha)*double(T)/double(size));
        atomicAdd(row+4,finite_linear+finite_q);atomicAdd(row+5,finite_q);
        atomicAdd(row+6,raw_alpha>.99f?1.:0.);atomicAdd(row+7,1.);
    }
    float error=0.f;
    for (int ch=0;ch<3;++ch)error=fmaxf(error,fabsf(tail[ch]-rgb[ch*size+pixel]));
    audit[pixel]=error;audit[size+pixel]=fabsf(T-1.f);
}
}

std::vector<torch::Tensor> analyze_slots_cuda(
    torch::Tensor colors, torch::Tensor geometry, torch::Tensor binning,
    torch::Tensor image_state, torch::Tensor raw_rgb, torch::Tensor gt,
    torch::Tensor image_gradient, torch::Tensor background, int64_t rendered) {
    TORCH_CHECK(colors.is_cuda(),"CUDA colors required");
    const auto device=colors.device();c10::cuda::CUDAGuard guard(device);
    for (const auto& t:{colors,geometry,binning,image_state,raw_rgb,gt,image_gradient,background}) {
        TORCH_CHECK(t.is_cuda() && t.device()==device && t.is_contiguous(),"Aligned contiguous CUDA tensors required");
    }
    TORCH_CHECK(colors.scalar_type()==torch::kFloat32 && colors.dim()==2 && colors.size(1)==3,"Precomputed float32 RGB colors required");
    TORCH_CHECK(raw_rgb.scalar_type()==torch::kFloat32 && raw_rgb.dim()==3 && raw_rgb.size(0)==3,"Raw CHW RGB required");
    TORCH_CHECK(gt.scalar_type()==torch::kFloat32 && image_gradient.scalar_type()==torch::kFloat32 && gt.sizes()==raw_rgb.sizes() && image_gradient.sizes()==raw_rgb.sizes(),"GT and pixel gradient shapes differ");
    TORCH_CHECK(background.scalar_type()==torch::kFloat32 && background.numel()==3,"RGB background required");
    for (const auto& t:{geometry,binning,image_state})TORCH_CHECK(t.scalar_type()==torch::kUInt8 && t.dim()==1,"Opaque uint8 state buffers required");
    const int64_t points=colors.size(0),height=raw_rgb.size(1),width=raw_rgb.size(2);
    TORCH_CHECK(points>0 && points<=INT_MAX && height>0 && width>0 && height*width<=INT_MAX && rendered>=0 && rendered<=INT_MAX,"Unsupported empty/oversized frame");
    const size_t P=size_t(points),N=size_t(height*width);
    // Parse only allocation prefixes preceding CUB workspaces. No dependency
    // on the CUB scan/sort scratch-size queries or original CUDA symbols.
    char* ptr=reinterpret_cast<char*>(geometry.data_ptr<uint8_t>());
    const char* end=ptr+geometry.numel();
    take<float>(ptr,P,end);take<bool>(ptr,P*3,end);take<int>(ptr,P,end);
    const float2* means=take<float2>(ptr,P,end);take<float>(ptr,P*6,end);
    const float4* opacity=take<float4>(ptr,P,end);
    ptr=reinterpret_cast<char*>(image_state.data_ptr<uint8_t>());end=ptr+image_state.numel();
    const float* trans=take<float>(ptr,N,end);
    const uint32_t* contributors=take<uint32_t>(ptr,N,end);
    const uint2* ranges=take<uint2>(ptr,N,end);
    ptr=reinterpret_cast<char*>(binning.data_ptr<uint8_t>());end=ptr+binning.numel();
    const uint32_t* list=take<uint32_t>(ptr,size_t(rendered),end);
    auto output=torch::zeros({points,8},colors.options().dtype(torch::kFloat64));
    auto audit=torch::zeros({2,height,width},colors.options());
    auto invalid=torch::zeros({3},colors.options().dtype(torch::kInt32));
    moments<<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,at::cuda::getCurrentCUDAStream()>>>(
        means,opacity,colors.data_ptr<float>(),list,ranges,trans,contributors,
        raw_rgb.data_ptr<float>(),gt.data_ptr<float>(),image_gradient.data_ptr<float>(),background.data_ptr<float>(),
        int(points),int(rendered),int(width),int(height),output.data_ptr<double>(),audit.data_ptr<float>(),invalid.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,audit,invalid};
}
