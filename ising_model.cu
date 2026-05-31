typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#include<cooperative_groups.h>
#include <cuda_pipeline_primitives.h>


typedef struct  {
    uint64_t state;
    uint64_t inc;
}pcg32_state;

__device__ uint32_t pcg32_random_r(pcg32_state *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 0x5851f42d4c958127U + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ void pcg32_init(pcg32_state *rng, uint64_t seed, uint64_t seq_id) {
    rng->state = 0U;
    rng->inc = (seq_id << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += seed;
    pcg32_random_r(rng);
}


__device__ __forceinline__ uint32_t wang_hash(uint32_t seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}



__device__ uint32_t get_prob_mask_pcg(pcg32_state* state,float prob,int precision){
    uint32_t prob_bin = (uint32_t)(prob*(1<<precision));
    uint32_t result = 0x00000000U;
    uint32_t unresolved=0xFFFFFFFFU;
  //  #pragma unroll
    for (int i = (1<<(precision-1));i>=1;i>>=1){
        uint32_t rand_mask = pcg32_random_r(state);
        if ((i & prob_bin)==i){
            result |= (unresolved&rand_mask);
            unresolved &= ~rand_mask;
        }else{
            unresolved &= rand_mask;
        }
        if (unresolved==0x00000000U) break;
    }
    return result;
}

extern "C" __global__ void initrng(uint64_t* state,int x_words_num,int y_lines){
int x_wordid = blockDim.x * blockIdx.x + threadIdx.x;
int y_lineid = blockDim.y * blockIdx.y + threadIdx.y;
bool in_bounds = (x_wordid < x_words_num && y_lineid < y_lines);
if (!in_bounds) return;
int s = y_lineid * x_words_num + x_wordid+1;
uint32_t seed_q = wang_hash((uint32_t)s *114514U);
if(seed_q == 0) seed_q = 0x12345678U;
pcg32_state pcg_rng;
pcg32_init(&pcg_rng,(uint64_t)seed_q,(uint64_t)s);
state[s-1] = pcg_rng.state;
// inc[s-1] = pcg_rng.inc;
}





extern "C"
__global__ void isingstep(
uint32_t* spin_field,
int* totEnergy,
bool red_or_black,
int x_words_num,
int y_lines,
float temperature,
float J,
int prec,int frames,uint64_t* state
){

int energy = 0;
int x_wordid = blockDim.x * blockIdx.x + threadIdx.x;
int y_lineid = blockDim.y * blockIdx.y + threadIdx.y;
bool in_bounds = (x_wordid < x_words_num && y_lineid < y_lines);
// if (x_wordid>=x_words_num || y_lineid >= y_lines) return;

__shared__ int energy_buffer[32];



if (in_bounds){

    
int s = y_lineid * x_words_num + x_wordid+1;
pcg32_state pcg_rng;
pcg_rng.state = state[s-1];
pcg_rng.inc = ((uint64_t)s << 1u) | 1u; 
float prob4 = expf(-4.0f*J/temperature);
float prob8 = prob4 * prob4;
uint32_t valid_mask = (red_or_black ^ (y_lineid%2==0)) ? 0x55555555U : 0xAAAAAAAAU;
uint32_t self_word = spin_field[y_lineid * x_words_num + x_wordid];
uint32_t up_neighbor = y_lineid == 0 ? spin_field[x_words_num*(y_lines-1)+x_wordid] : spin_field[x_words_num*(y_lineid-1)+x_wordid];
uint32_t down_neighbor = y_lineid == y_lines - 1 ? spin_field[x_wordid] : spin_field[x_words_num*(y_lineid+1)+x_wordid];
uint32_t left_neighbor = x_wordid == 0 ? spin_field[(y_lineid+1) * x_words_num - 1] : spin_field[x_words_num*y_lineid+x_wordid-1];
uint32_t right_neighbor = x_wordid == x_words_num-1 ? spin_field[(y_lineid) * x_words_num] : spin_field[x_words_num*y_lineid+x_wordid+1];
left_neighbor = (self_word << 1) | (left_neighbor >> 31);
right_neighbor = (self_word >> 1) | (right_neighbor << 31);
up_neighbor = ~(self_word ^ up_neighbor);
down_neighbor = ~(self_word ^ down_neighbor);
left_neighbor = ~(self_word ^ left_neighbor);
right_neighbor = ~(self_word ^ right_neighbor);
uint32_t same_bonds = 
    __popc(up_neighbor & valid_mask) +
    __popc(down_neighbor & valid_mask) +
    __popc(left_neighbor & valid_mask) +
    __popc(right_neighbor & valid_mask);


energy = 64 - 2 * (int)same_bonds;
// atomicAdd(totEnergy,(float)energy);
uint32_t and1 = up_neighbor & down_neighbor;
uint32_t and2 = left_neighbor & right_neighbor;
uint32_t xor1 = up_neighbor ^ down_neighbor;
uint32_t xor2 = left_neighbor ^ right_neighbor;
uint32_t mask_4 = and1 & and2;
uint32_t mask_3 = (and1 & xor2) | (and2 & xor1);
uint32_t mask_le_2 = ~(mask_4 | mask_3);
uint32_t randnum_4 = get_prob_mask_pcg(&pcg_rng,prob4,prec);
uint32_t randnum_8 = get_prob_mask_pcg(&pcg_rng,prob8,prec);
uint32_t flip_or_not = (mask_4 & randnum_8) | (mask_3 & randnum_4) | mask_le_2;
flip_or_not = flip_or_not & valid_mask;
self_word = flip_or_not ^ self_word;
spin_field[y_lineid * x_words_num + x_wordid] = self_word;

state[s-1] = pcg_rng.state;
}   

#pragma unroll
for (int i=16;i>=1;i>>=1){
    energy+=__shfl_down_sync(0xFFFFFFFFU,energy,i);
}
int tid = threadIdx.y * blockDim.x + threadIdx.x;
int lane = tid % 32;
int wid = tid / 32;
int shared_num = (blockDim.x * blockDim.y + 31) / 32;
if (lane==0){
    energy_buffer[wid] = energy;
}
__syncthreads();
if (wid==0){
    energy = lane<shared_num ? energy_buffer[lane] : 0;
    for (int i=16;i>=1;i>>=1){
        energy+=__shfl_down_sync(0xFFFFFFFFU,energy,i);
    }
    if (lane==0){
        atomicAdd(totEnergy,energy);
    }
}

}

extern "C"
__global__ void render_spin_field(
    const uint32_t* spin_field,
    unsigned char* image_buffer,
    int x_words_num,
    int y_lines
){
    int px = blockDim.x * blockIdx.x + threadIdx.x;
    int py = blockDim.y * blockIdx.y + threadIdx.y;
    
    int width = x_words_num * 32;
    int pid = py * width + px;

    if (px >= width || py >= y_lines) return;

    int word_idx = px / 32;
    int bit_idx = px % 32; 

    uint32_t word = __ldg(&spin_field[py * x_words_num + word_idx]);

    bool is_up = (word >> bit_idx) & 1;
    uint32_t color = is_up ? 0xFFFFFFFF : 0xFF000000;
    // 朝上涂白(255)，朝下涂黑(0)
    ((uint32_t*)image_buffer)[pid] = color;
}