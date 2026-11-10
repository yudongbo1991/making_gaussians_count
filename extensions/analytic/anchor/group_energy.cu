/* Read-only local anchor energy. Raster state layout and alpha-over traversal
 * follow Inria / GRAPHDECO diff-gaussian-rasterization (2023); see LICENSE.md.
 * Group local opacity directions before squaring. This is not a finite group
 * deletion renderer or a claim that a diagonal image Hessian is novel.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <vector>

namespace {
template<class T> T* take(char*& ptr,size_t count,const char* end) {
    uintptr_t p=(reinterpret_cast<uintptr_t>(ptr)+127)&~uintptr_t(127);
    T* result=reinterpret_cast<T*>(p);ptr=reinterpret_cast<char*>(result+count);
    TORCH_CHECK(ptr<=end,"Renderer state buffer prefix too short");return result;
}

struct Ray {
    const float2* xy;const float4* conic;const float* colors;const uint32_t* list;
    const float* rgb;int P,x,y,pixel,size;int64_t at,first;
    float T,tail[3];bool bad;
    __device__ Ray(const float2* a,const float4* b,const float* c,const uint32_t* d,
        const float* image,const float* bg,int points,int xx,int yy,int pix,int pixels,
        uint2 range,uint32_t last,float trans):xy(a),conic(b),colors(c),list(d),rgb(image),
        P(points),x(xx),y(yy),pixel(pix),size(pixels),at(int64_t(range.x)+last-1),first(range.x),T(trans),bad(false) {
        for(int ch=0;ch<3;++ch)tail[ch]=bg[ch];
    }
    __device__ bool next(uint32_t& id,double* direction) {
        while(at>=first) {
            id=list[at--];if(id>=uint32_t(P)){bad=true;return false;}
            const float2 center=xy[id];const float4 c=conic[id];
            const float dx=center.x-float(x),dy=center.y-float(y);
            const float power=-.5f*(c.x*dx*dx+c.z*dy*dy)-c.y*dx*dy;
            if(power>0.f)continue;
            const float raw=c.w*expf(power),alpha=fminf(.99f,raw);
            if(alpha<1.f/255.f)continue;
            T=T/(1.f-alpha);
            for(int ch=0;ch<3;++ch) {
                const float color=colors[id*3+ch],base=rgb[ch*size+pixel];
                direction[ch]=(base>=0.f && base<=1.f)?double(raw)*double(T)*double(color-tail[ch]):0.;
                tail[ch]=alpha*color+(1.f-alpha)*tail[ch];
            }
            return true;
        }
        return false;
    }
};

template<int CAP> __global__ void grouped(
    const float2* xy,const float4* conic,const float* colors,const uint32_t* list,
    const uint2* ranges,const float* final_T,const uint32_t* n_contrib,
    const float* rgb,const float* bg,const int64_t* parents,const int* selected,
    int P,int R,int W,int H,double* energy,float* audit,int* stats,int* invalid) {
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=W || y>=H)return;
    const int pixel=y*W+x,size=W*H;if(!selected[pixel])return;
    const uint2 range=ranges[blockIdx.y*((W+15)/16)+blockIdx.x];
    if(range.x>range.y || range.y>uint32_t(R)){atomicAdd(invalid,1);return;}
    const uint32_t last=n_contrib[pixel];
    if(last>range.y-range.x){atomicAdd(invalid+1,1);return;}
    int keys[CAP];double sums[CAP][3];
    for(int k=0;k<CAP;++k){keys[k]=-1;for(int c=0;c<3;++c)sums[k][c]=0.;}
    Ray ray(xy,conic,colors,list,rgb,bg,P,x,y,pixel,size,range,last,final_T[pixel]);
    uint32_t id;double direction[3];int occupied=0,count=0;bool overflow=false;
    while(ray.next(id,direction)) {
        ++count;if(overflow)continue;
        const int key=int(parents[id]);unsigned at=(uint32_t(key)*2654435761U)%CAP;
        int probes=0;
        for(;probes<CAP;++probes){if(keys[at]==-1 || keys[at]==key)break;at=(at+1)%CAP;}
        if(probes==CAP){overflow=true;continue;}
        if(keys[at]==-1){keys[at]=key;++occupied;}
        for(int c=0;c<3;++c)sums[at][c]+=direction[c];
    }
    if(ray.bad){atomicAdd(invalid+2,1);return;}
    float error=0.f;for(int c=0;c<3;++c)error=fmaxf(error,fabsf(ray.tail[c]-rgb[c*size+pixel]));
    audit[pixel]=error;audit[size+pixel]=fabsf(ray.T-1.f);
    stats[pixel]=count;stats[size+pixel]=overflow?CAP+1:occupied;stats[2*size+pixel]=1;
    const double norm=1./(3.*double(size));
    if(!overflow) {
        for(int k=0;k<CAP;++k)if(keys[k]>=0) {
            double q=0.;for(int c=0;c<3;++c)q+=sums[k][c]*sums[k][c];
            atomicAdd(energy+keys[k],q*norm);
        }
        return;
    }
    // No partial fast-path sum has been committed. Exact fallback uses at most
    // CAP distinct IDs per pass. Capacity controls work, never approximation.
    int lower=-1;
    while(true) {
        int used=0,maximum=-1;
        Ray choose(xy,conic,colors,list,rgb,bg,P,x,y,pixel,size,range,last,final_T[pixel]);
        while(choose.next(id,direction)) {
            int key=int(parents[id]);maximum=max(maximum,key);if(key<=lower)continue;
            int at=0;while(at<used && keys[at]<key)++at;
            if(at<used && keys[at]==key)continue;
            if(at==CAP)continue;
            if(used<CAP)++used;
            for(int j=used-1;j>at;--j)keys[j]=keys[j-1];keys[at]=key;
        }
        ++stats[2*size+pixel];
        if(choose.bad){atomicAdd(invalid+2,1);return;}
        if(!used)break;
        for(int k=0;k<used;++k)for(int c=0;c<3;++c)sums[k][c]=0.;
        Ray accumulate(xy,conic,colors,list,rgb,bg,P,x,y,pixel,size,range,last,final_T[pixel]);
        while(accumulate.next(id,direction)) {
            int key=int(parents[id]),lo=0,hi=used;
            while(lo<hi){int mid=(lo+hi)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
            if(lo<used && keys[lo]==key)for(int c=0;c<3;++c)sums[lo][c]+=direction[c];
        }
        ++stats[2*size+pixel];
        if(accumulate.bad){atomicAdd(invalid+2,1);return;}
        for(int k=0;k<used;++k) {
            double q=0.;for(int c=0;c<3;++c)q+=sums[k][c]*sums[k][c];
            atomicAdd(energy+keys[k],q*norm);
        }
        lower=keys[used-1];if(lower>=maximum)break;
    }
}
}

std::vector<torch::Tensor> analyze_anchor_groups_cuda(torch::Tensor colors,torch::Tensor geometry,
    torch::Tensor binning,torch::Tensor image_state,torch::Tensor rgb,torch::Tensor background,
    torch::Tensor parents,torch::Tensor selected,int64_t rendered,int64_t anchors,int64_t capacity) {
    TORCH_CHECK(colors.is_cuda(),"CUDA colors required");const auto device=colors.device();
    c10::cuda::CUDAGuard guard(device);
    for(const auto& t:{colors,geometry,binning,image_state,rgb,background,parents,selected})
        TORCH_CHECK(t.is_cuda() && t.device()==device && t.is_contiguous(),"Contiguous same-device CUDA inputs required");
    TORCH_CHECK(colors.scalar_type()==torch::kFloat32 && colors.dim()==2 && colors.size(1)==3,"Float32 RGB colors required");
    TORCH_CHECK(rgb.scalar_type()==torch::kFloat32 && rgb.dim()==3 && rgb.size(0)==3,"Float32 CHW RGB required");
    TORCH_CHECK(background.scalar_type()==torch::kFloat32 && background.numel()==3,"RGB background required");
    const int64_t P=colors.size(0),H=rgb.size(1),W=rgb.size(2),N=H*W;
    TORCH_CHECK(P>0 && P<=INT_MAX && H>0 && W>0 && N<=INT_MAX && rendered>=0 && rendered<=INT_MAX && anchors>0 && anchors<INT_MAX,"Unsupported frame size");
    TORCH_CHECK(parents.scalar_type()==torch::kInt64 && parents.dim()==1 && parents.numel()==P,"One int64 parent ID per emitted slot required");
    TORCH_CHECK(selected.scalar_type()==torch::kInt32 && selected.dim()==2 && selected.size(0)==H && selected.size(1)==W,"Int32 HxW pixel selector required");
    TORCH_CHECK(parents.min().item<int64_t>()>=0 && parents.max().item<int64_t>()<anchors,"Parent ID out of range");
    for(const auto& t:{geometry,binning,image_state})TORCH_CHECK(t.scalar_type()==torch::kUInt8 && t.dim()==1,"Opaque byte buffers required");
    char* ptr=reinterpret_cast<char*>(geometry.data_ptr<uint8_t>());const char* end=ptr+geometry.numel();
    take<float>(ptr,P,end);take<bool>(ptr,P*3,end);take<int>(ptr,P,end);
    const float2* xy=take<float2>(ptr,P,end);take<float>(ptr,P*6,end);const float4* conic=take<float4>(ptr,P,end);
    ptr=reinterpret_cast<char*>(image_state.data_ptr<uint8_t>());end=ptr+image_state.numel();
    const float* trans=take<float>(ptr,N,end);const uint32_t* contributors=take<uint32_t>(ptr,N,end);const uint2* ranges=take<uint2>(ptr,N,end);
    ptr=reinterpret_cast<char*>(binning.data_ptr<uint8_t>());end=ptr+binning.numel();const uint32_t* list=take<uint32_t>(ptr,rendered,end);
    auto energy=torch::zeros({anchors},colors.options().dtype(torch::kFloat64));
    auto audit=torch::zeros({2,H,W},colors.options());
    auto stats=torch::zeros({3,H,W},colors.options().dtype(torch::kInt32));
    auto invalid=torch::zeros({3},colors.options().dtype(torch::kInt32));
    const dim3 grid((W+15)/16,(H+15)/16),block(16,16);const auto stream=at::cuda::getCurrentCUDAStream();
#define LAUNCH(C) grouped<C><<<grid,block,0,stream>>>(xy,conic,colors.data_ptr<float>(),list,ranges,trans,contributors,rgb.data_ptr<float>(),background.data_ptr<float>(),parents.data_ptr<int64_t>(),selected.data_ptr<int>(),int(P),int(rendered),int(W),int(H),energy.data_ptr<double>(),audit.data_ptr<float>(),stats.data_ptr<int>(),invalid.data_ptr<int>())
    if(capacity==1){LAUNCH(1);}else if(capacity==2){LAUNCH(2);}else if(capacity==8){LAUNCH(8);}else if(capacity==64){LAUNCH(64);}else TORCH_CHECK(false,"Supported diagnostic capacities1,2,8,64 only");
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {energy,audit,stats,invalid};
}
