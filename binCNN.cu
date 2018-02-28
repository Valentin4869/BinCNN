#include <cuda_runtime.h>
#include "GeneralUtils.h"
#include <stdio.h>	
#include <cuda_profiler_api.h>
#include <math.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cudnn.h>


#define FRAMES 1000
#define REPEAT 1

//#define MEASURE_MEMCPY

//#define FUSED_CONVOLUTION //1:3
//#define BINARIZED_INPUT // 
//#define NO_SHARED_MEMORY
#define USE_CUBLAS


#ifdef BINARIZED_INPUT
#define TEST_LOOP \
		for(int r=0; r<REPEAT; r++)\
			for (int i = 0;\
				i < FRAMES;\
				i++, normalizeImage(normalizedIm, (int*)&batch[(i-1) * 96 * 96 * 3], -127, 127)) // first input is the true image; after that we're feeding it junk

#else

#define TEST_LOOP \
		for(int r=0; r<REPEAT; r++)\
			for (int i = 0;\
				i < FRAMES;\
				i++) 

#endif


//FIXED PARAMETERS
#define WIDTH 96
#define HEIGHT 96
#define C1 3
#define C2 32
#define KERNEL_RADIUS 2
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1) 
#define KERNEL_SIZE (KERNEL_LENGTH*KERNEL_LENGTH) //=25
#define D1_BLOCK_DIM_X 8
#define D1_BLOCK_DIM_Y 100
#define D1_BLOCK_DIM 64
#define D1_BLOCK_LOG2 6
#define D1_PARTS 9 //=((24*24)/D1_BLOCK)
#define CONV_BLOCK_DIM_Y 32
#define CONV_BLOCK_DIM_X 32
#define FMAPS1 32
#define MP1_BLOCK_DIM 32
#define MP2_BLOCK_DIM 16
#define IM2COL_BLOCK_DIM_X 96
#define IM2COL_BLOCK_DIM_Y 2
#define IM2COL_REGION_3D ((IM2COL_BLOCK_DIM_X+ 2*KERNEL_RADIUS)*(IM2COL_BLOCK_DIM_Y+ 2*KERNEL_RADIUS)*C1) // =((96+ 2*2)*(2+ 2*2)) 
#define IM2COL_REGION_2D ((IM2COL_BLOCK_DIM_X+ 2*KERNEL_RADIUS)*(IM2COL_BLOCK_DIM_Y+ 2*KERNEL_RADIUS)) // =((96+ 2*2)*(2+ 2*2)*3) 
#define IM2COL_2_BLOCK_DIM_X 48
#define IM2COL_2_BLOCK_DIM_Y 4
#define IM2COL_2_REGION_2D ((IM2COL_2_BLOCK_DIM_X+ 2*KERNEL_RADIUS)*(IM2COL_2_BLOCK_DIM_Y+ 2*KERNEL_RADIUS)) // =(8+4)x(48+4)
#define IM2COL_2_REGION_3D ((IM2COL_2_BLOCK_DIM_X+ 2*KERNEL_RADIUS)*(IM2COL_2_BLOCK_DIM_Y+ 2*KERNEL_RADIUS)*C2) // =(8+4)x(48+4)x32
#define PACK_BITWIDTH 25
#define D1_PACK_BITWIDTH 32
#define TOTAL_THREADS 1024 // define using block dims later
//#define CHUNKS1 int(ceilf(float(IM2COL_BLOCK_DIM_Y * IM2COL_BLOCK_DIM_X * ROW_LENGTH) / float(TOTAL_THREADS)))
//#define DIV(a,b) (int(floorf(float(a)/float(b)))) // This needs to be eliminated/replaced
#define DIV(a,b) (int)__fdividef(a,b)
//VARIABLE PARAMETERS
//these are chosen to maximize occupancy on the GTX 1080
#define CONV_NUM_CHUNKS 1 // how many chunks of 32x32 to compute in output per block; alternatively, how many chunks of 3x32 to load from input
#define CONV2_NUM_CHUNKS 2 // how many chunks of 32x32 to compute in output per block; alternatively, how many chunks of 32x32 to load from input




//FOR DEBUGGING:
//#define VERIFY 1

#define UNIQUE_BLOCK(idx,idy,idz) (blockIdx.x==idx && blockIdx.y==idy&& blockIdx.z==idz)
#define UNIQUE_THREAD(idx,idy,idz) (threadIdx.x ==idx && threadIdx.y==idy&& threadIdx.z==idz)
#define T(THREADIDX, s,...) if(UNIQUE_BLOCK(0,0,0) && UNIQUE_THREAD(THREADIDX,0,0)){printf(s,__VA_ARGS__);}
#define CUDA_ERROR(RET, OPNAME)	if (RET != cudaSuccess) {\
fprintf(stderr, "%s failed: %s\n" , OPNAME, cudaGetErrorString(RET)); pause(); return -1; }

#define CUDNN_ERROR(status, opname)                               \
        if(status!=CUDNN_STATUS_SUCCESS) {                                                   \
   fprintf(stderr, "%s failed at line %i: %s\n", opname,__LINE__, cudnnGetErrorString(status)); \
	pause();}

/*	if (UNIQUE_BLOCK(0, 0, 0) && UNIQUE_THREAD(0, 0, 0))
printf("\n");

if (UNIQUE_BLOCK(0, 0, 0) && UNIQUE_THREAD(0, 0, 0))
for (int i = 0; i < 3; i++) {
for (int j = 0; j < 32; j++)
{
printf("%i ", sh_im[i * 32 + j]);
}
printf("\n");
}*/

void pause() {

#ifdef _WIN32
	system("pause");
#elif __linux__
	//system("read -p \"Press any key to continue...\n\"");
#endif
}




#ifdef BINARIZED_INPUT
//packed conv1 and conv2 weights
__constant__ unsigned int wgtConv1[FMAPS1 * KERNEL_SIZE*C1 / PACK_BITWIDTH];
//__constant__ unsigned int wgtConv2[FMAPS1 * KERNEL_SIZE*C2 / PACK_BITWIDTH]; //not used

#else
__constant__  int wgtConv1[FMAPS1 * KERNEL_SIZE*C1];

__constant__ int T[] = {-127,-100,-70};

#endif




#ifdef FUSED_CONVOLUTION

	
//Receives 96x96x3 input
//

//<<<(48,1,3),(96,2)>>>
__global__ void fIm2Col3d(float *src, float* dst)
{


	src += (blockIdx.x*IM2COL_BLOCK_DIM_Y) * IM2COL_BLOCK_DIM_X + blockIdx.z*(WIDTH  * HEIGHT) - (blockIdx.x>0)*KERNEL_RADIUS*IM2COL_BLOCK_DIM_X;
	dst += blockIdx.x * (IM2COL_BLOCK_DIM_Y * IM2COL_BLOCK_DIM_X) + blockIdx.z*(WIDTH  * HEIGHT * 25) + threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x;  // store as 32x(96*96)


	__shared__  float sh_block[IM2COL_REGION_2D];

	for (int i = 0; i<4; i++)
		if (threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x + i*IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y< IM2COL_REGION_2D)
			sh_block[threadIdx.y * IM2COL_BLOCK_DIM_X + threadIdx.x + i * IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y] = 0;


	__syncthreads();


	//This should probably be a loop
	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y)] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH)];//y 


	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * 2] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH) + IM2COL_BLOCK_DIM_X * 2];//y 

	if (blockIdx.x>0 && blockIdx.x < gridDim.x - 1)
		sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + 4 * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)]
		= src[(threadIdx.x) // x
		+ (threadIdx.y*WIDTH) + 4 * IM2COL_BLOCK_DIM_X];




	__syncthreads();







	unsigned int offset = (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) *threadIdx.y + (threadIdx.x);

	for (int i = 0, k = 0; i < PACK_BITWIDTH; i++)
	{

		//calculate div(i,KERNEL_LENGTH)
		if (i - k * KERNEL_LENGTH == KERNEL_LENGTH)
			k++;

		dst[i*WIDTH*HEIGHT] = sh_block[offset + k * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) + (i - k * KERNEL_LENGTH)];	// store as (25*3)x(96*96)


	}





}
__global__ void fxnorConv1(const float*  src, float* dst, float*  wgt) {



	const  int idxLin = threadIdx.y*CONV_BLOCK_DIM_X + threadIdx.x; //local linear index

																	//src += blockIdx.x*(CONV_NUM_CHUNKS*blockDim.x * KERNEL_SIZE*C1 / PACK_BITWIDTH); //each block processes a (32*4)x3 region from src
																	//dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

																	// src is 32x(96*96)
	src += blockIdx.x*(CONV_NUM_CHUNKS * CONV_BLOCK_DIM_X);
	dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

	//	__shared__ float sh_wgt[FMAPS1 * 75];
	__shared__  float sh_im[C1 * 25 * CONV_BLOCK_DIM_X];




	for (int j = 0; j < CONV_NUM_CHUNKS; j++) {

		float sumC = 0;

		//if (threadIdx.y<3)
		sh_im[idxLin] = src[j* CONV_BLOCK_DIM_X + threadIdx.y*WIDTH  * HEIGHT + threadIdx.x];
		sh_im[idxLin + 32 * 32] = src[j* CONV_BLOCK_DIM_X + (32 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];

		if (threadIdx.y < 11)
			sh_im[idxLin + 32 * 32 * 2] = src[j* CONV_BLOCK_DIM_X + (32 * 2 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];


		__syncthreads();




		for (int i = 0; i < 25 * 3; i++) {
			//sumC += PACK_BITWIDTH - 2 * (__popc(wgtConv1[threadIdx.y *C1 + i] ^ sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i]));
			sumC += wgt[threadIdx.y * 25 * 3 + i] * sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i];
			//	sumC += wgt[threadIdx.y * 25 * 3 + i] * src[j* CONV_BLOCK_DIM_X +  threadIdx.x + WIDTH  * HEIGHT *i];



		}



		dst[threadIdx.y * WIDTH * HEIGHT + CONV_BLOCK_DIM_X * j + threadIdx.x] = (sumC);

		__syncthreads();

	}


}

	//Each thread extracts a column as usual, then divide up the wgts over the threads, each thread computes a pixel for each of 4 featuremaps. For each of my columns, compute with my allocated wgts (4)


#else //separate im2col and gemm kernels




//<<<(24,1,3),(96,2)>>>
__global__ void bIm2Col3d(int *src, unsigned int* pkDst)
{


	int locT = T[blockIdx.z];

	src += (blockIdx.x*IM2COL_BLOCK_DIM_Y) * IM2COL_BLOCK_DIM_X + blockIdx.z*(WIDTH  * HEIGHT) - (blockIdx.x>0)*KERNEL_RADIUS*IM2COL_BLOCK_DIM_X;
	//pkDst += blockIdx.x * (IM2COL_2_BLOCK_DIM_Y * IM2COL_2_BLOCK_DIM_X) * (C2*KERNEL_SIZE / PACK_BITWIDTH) +blockIdx.z; //store as (48*48)x32
	pkDst += blockIdx.x * (IM2COL_BLOCK_DIM_Y * IM2COL_BLOCK_DIM_X) + blockIdx.z*(WIDTH  * HEIGHT);  // store as 32x(48*48)


	__shared__  int sh_block[IM2COL_REGION_2D];

	for (int i = 0; i<4; i++)
		if (threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x + i*IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y< IM2COL_REGION_2D)
			sh_block[threadIdx.y * IM2COL_BLOCK_DIM_X + threadIdx.x + i * IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y] = 0;


	__syncthreads();


	//This should probably be a loop
	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y)] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH)];//y 


	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * 2] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH) + IM2COL_BLOCK_DIM_X * 2];//y 

	if (blockIdx.x>0 && blockIdx.x < gridDim.x - 1)
		sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + 4 * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)]
		= src[(threadIdx.x) // x
		+ (threadIdx.y*WIDTH) + 4 * IM2COL_BLOCK_DIM_X];




	__syncthreads();




	//Packing stage

	unsigned int v = 0;
	unsigned int offset = (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) *threadIdx.y + (threadIdx.x);

	for (int i = 0, k = 0; i < PACK_BITWIDTH; i++)
	{

		//calculate div(i,KERNEL_LENGTH)
		if (i - k * KERNEL_LENGTH == KERNEL_LENGTH)
			k++;

		unsigned int sign;
		unsigned int idx = offset + k * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) + (i - k * KERNEL_LENGTH); //==(i % KERNEL_LENGTH)


																											//sign = __vsetgts4(sh_block[idx], 0);
																											//sign = __vcmpgts4(sh_block[idx], 0);
		sign = sh_block[idx]+locT>0;
		v |= (sign << (PACK_BITWIDTH - 1 - i));



	}




	//pkDst[threadIdx.x*32] = v; //store as (48*48)x32
	pkDst[threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x] = v;	// store as 3x(96*96)



























}


// wgt is 32x3
// src is 3x(96*96)
// dst is 32x(96*96)
// Function of each block:
//		loads the entire weights matrix to shared memory
//		loads a chunk of 3x32, computes 32x32 result, and load next chunk... repeat. So each block computes a 32x(3*32) region in the output. Note: 32x32 (in otutput) is one row of length 32 for each of 32 channels,
//
//<<< 72, (32,32) >>>		
__global__ void bxnorConv1(const unsigned  int* src, int* dst, unsigned int* wgt) {



	const int idxLin = threadIdx.y*blockDim.x + threadIdx.x; //local linear index

															 //src += blockIdx.x*(CONV_NUM_CHUNKS*blockDim.x * KERNEL_SIZE*C1 / PACK_BITWIDTH); //each block processes a (32*4)x3 region from src
															 //dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

															 // src is 32x(96*96)
	src += blockIdx.x*(CONV_NUM_CHUNKS * CONV_BLOCK_DIM_X);
	dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

	/*__shared__ unsigned int sh_wgt[FMAPS1 * KERNEL_SIZE1];*/
	__shared__ unsigned int sh_im[C1 * CONV_BLOCK_DIM_X];




	for (int j = 0; j < CONV_NUM_CHUNKS; j++) {

		int sumC = 0;

		if (threadIdx.y<3)
			sh_im[idxLin] = src[j* blockDim.x + threadIdx.y*WIDTH  * HEIGHT + threadIdx.x];

		__syncthreads();




		for (int i = 0; i < C1; i++) {
			sumC += PACK_BITWIDTH - 2 * (__popc(wgtConv1[threadIdx.y *C1 + i] ^ sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i]));


		}



		dst[threadIdx.y * WIDTH * HEIGHT + CONV_BLOCK_DIM_X * j + threadIdx.x] = (sumC);

		__syncthreads();

	}


}






//<<< (3,3,32) , (32,32) >>>
__global__ void bMaxPool1(const int* src, int* dst) {

	dst += blockIdx.x * blockDim.x / 2 + blockIdx.y * WIDTH / 2 * blockDim.y / 2 + blockIdx.z * WIDTH / 2 * WIDTH / 2;
	src += blockIdx.x * blockDim.x + blockIdx.y*blockDim.y * WIDTH + blockIdx.z * WIDTH*HEIGHT;



	__shared__ int sh_data[MP1_BLOCK_DIM][MP1_BLOCK_DIM];

	sh_data[threadIdx.y][threadIdx.x] = src[threadIdx.y * WIDTH + threadIdx.x];


	__syncthreads();


	if (threadIdx.x < MP1_BLOCK_DIM / 2 && threadIdx.y < MP1_BLOCK_DIM / 2) {

		int locMax = sh_data[threadIdx.y * 2][threadIdx.x * 2];

		if (locMax < sh_data[threadIdx.y * 2][threadIdx.x * 2 + 1])
			locMax = sh_data[threadIdx.y * 2][threadIdx.x * 2 + 1];




		if (locMax < sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2])
			locMax = sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2];

		if (locMax < sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1])
			locMax = sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1];



		dst[threadIdx.y * WIDTH / 2 + threadIdx.x] = locMax;




		//int locMax1 = __vmaxs2(sh_data[threadIdx.y * 2][threadIdx.x * 2 ], sh_data[threadIdx.y * 2][threadIdx.x * 2 + 1]);

		//int locMax2 = __vmaxs2(sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2], sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1]);

		//dst[threadIdx.y * WIDTH / 2 + threadIdx.x] = __vmaxs2(locMax1, locMax2);



	}






}


//<<<(24,1,32),(48,2)>>>
__global__ void bIm2Col3d_2(int *src, unsigned int* pkDst)
{
	
	src += (blockIdx.x*IM2COL_2_BLOCK_DIM_Y) * IM2COL_2_BLOCK_DIM_X + blockIdx.z*(WIDTH / 2 * HEIGHT / 2) - (blockIdx.x>0)*KERNEL_RADIUS*IM2COL_2_BLOCK_DIM_X;
	//pkDst += blockIdx.x * (IM2COL_2_BLOCK_DIM_Y * IM2COL_2_BLOCK_DIM_X) * (C2*KERNEL_SIZE / PACK_BITWIDTH) +blockIdx.z; //store as (48*48)x32
	pkDst += blockIdx.x * (IM2COL_2_BLOCK_DIM_Y * IM2COL_2_BLOCK_DIM_X) + blockIdx.z*(WIDTH / 2 * HEIGHT / 2);  // store as 32x(48*48)


	__shared__  int sh_block[IM2COL_2_REGION_2D];

	for (int i = 0; i<4; i++)
		if (threadIdx.y*IM2COL_2_BLOCK_DIM_X + threadIdx.x + i*IM2COL_2_BLOCK_DIM_X *IM2COL_2_BLOCK_DIM_Y< IM2COL_2_REGION_2D)
			sh_block[threadIdx.y*IM2COL_2_BLOCK_DIM_X + threadIdx.x + i * IM2COL_2_BLOCK_DIM_X * IM2COL_2_BLOCK_DIM_Y] = 0;


	__syncthreads();



	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS
		+ (threadIdx.x) + ((IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y)]
		= src[(threadIdx.x) + (threadIdx.y*WIDTH / 2)];


	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS
		+ (threadIdx.x) + ((IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + (IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * 2]
		= src[(threadIdx.x)
		+ (threadIdx.y*WIDTH / 2) + IM2COL_2_BLOCK_DIM_X * 2];

	if (blockIdx.x>0 && blockIdx.x < gridDim.x - 1)
		sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS
		+ (threadIdx.x) + ((IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + 2 * 2 * (IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS)]
		= src[(threadIdx.x) + (threadIdx.y*WIDTH / 2) + 2 * 2 * IM2COL_2_BLOCK_DIM_X];




	__syncthreads();






	unsigned int v = 0;
	unsigned int offset = (IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) *threadIdx.y + (threadIdx.x);

	for (int i = 0, k = 0; i < PACK_BITWIDTH; i++)
	{

		//calculate div(i,KERNEL_LENGTH)
		if (i - k * KERNEL_LENGTH == KERNEL_LENGTH)
			k++;

		unsigned int sign;
		unsigned int idx = offset + k * (IM2COL_2_BLOCK_DIM_X + 2 * KERNEL_RADIUS) + (i - k * KERNEL_LENGTH); //==(i % KERNEL_LENGTH)


																											  //sign = __vsetgts4(sh_block[idx], 0);
		sign = sh_block[idx]>0;
		v |= (sign << (PACK_BITWIDTH - 1 - i));


	}



	//pkDst[ threadIdx.x*32] = v; //store as (48*48)x32
	pkDst[threadIdx.y*IM2COL_2_BLOCK_DIM_X + threadIdx.x] = v;	// store as 32x(48*48)





}




// src is (48*48)x32 or 32x48*48
// wgt is 32x32
// Function of each block:
//		loads the entire weights matrix to shared memory
//		loads a chunk of 32x32 from src, computes 32x32 result, and load next chunk... repeat 2 times. So each block computes a 32x(2*32) region in the output
//<<< 18 , (32,32) >>>
// 
__global__ void bxnorConv2(const unsigned  int* src, const unsigned int* wgt, int* dst) {



	const int idxLin = threadIdx.y*blockDim.x + threadIdx.x; //local linear index

															 //src is (48*48)x(32)
															 //src += blockIdx.x*(CONV2_NUM_CHUNKS*blockDim.x * C2*KERNEL_SIZE / PACK_BITWIDTH); 
															 //dst += blockIdx.x * CONV2_NUM_CHUNKS*CONV_BLOCK_DIM_X;															

															 // src is 32x(48*48)
	src += blockIdx.x*(CONV2_NUM_CHUNKS * CONV_BLOCK_DIM_X);
	dst += blockIdx.x * CONV2_NUM_CHUNKS*CONV_BLOCK_DIM_X;

	__shared__ unsigned int sh_wgt[FMAPS1 * C2*KERNEL_SIZE / PACK_BITWIDTH];
	__shared__ unsigned int sh_im[(C2*KERNEL_SIZE / PACK_BITWIDTH) * CONV_BLOCK_DIM_X];

	sh_wgt[idxLin] = wgt[idxLin];
	__syncthreads();

	// if src is 32x(48*48)
	for (int j = 0; j <CONV2_NUM_CHUNKS; j++) {

		int sumC = 0;

		sh_im[idxLin] = src[j* blockDim.x + threadIdx.y*WIDTH / 2 * HEIGHT / 2 + threadIdx.x];

		__syncthreads();


		for (int i = 0; i < C2; i++)
			sumC += PACK_BITWIDTH - 2 * (__popc(sh_wgt[threadIdx.y * C2 + i] ^ sh_im[threadIdx.x + C2*i]));

		dst[threadIdx.y * WIDTH / 2 * HEIGHT / 2 + CONV_BLOCK_DIM_X * j + threadIdx.x] = (sumC);
		__syncthreads();

	}



}




//<<< (3,3,32) , (16,16) >>>
__global__ void bMaxPool2(const int* src, int* dst) {

	dst += blockIdx.x * blockDim.x / 2 + blockIdx.y * WIDTH / 4 * blockDim.y / 2 + blockIdx.z * WIDTH / 4 * HEIGHT / 4;
	src += blockIdx.x * blockDim.x + blockIdx.y*blockDim.y * WIDTH / 2 + blockIdx.z *  WIDTH / 2 * HEIGHT / 2;



	__shared__ int sh_data[MP2_BLOCK_DIM][MP2_BLOCK_DIM];

	sh_data[threadIdx.y][threadIdx.x] = src[threadIdx.y * WIDTH / 2 + threadIdx.x];


	__syncthreads();


	if (threadIdx.x < MP2_BLOCK_DIM / 2 && threadIdx.y < MP2_BLOCK_DIM / 2) {

		int locMax = sh_data[threadIdx.y * 2][threadIdx.x * 2];

		if (locMax < sh_data[threadIdx.y * 2][threadIdx.x * 2 + 1])
			locMax = sh_data[threadIdx.y * 2][threadIdx.x * 2 + 1];

		if (locMax < sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2])
			locMax = sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2];

		if (locMax < sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1])
			locMax = sh_data[threadIdx.y * 2 + 1][threadIdx.x * 2 + 1];

		dst[threadIdx.y * WIDTH / 4 + threadIdx.x] = locMax;
	}






}




//This kernel packs column-wise to compensate for the flattening of the output of conv2
// TensorFlow flattens channel-wise (column-size for 32x(24*24) matrix for example), and here we're using TensorFlow weights (Note: weights are flipped).
//<<<1,24*24>>>
__global__ void packRowsDense1(const int *src, unsigned int *dst, int size)
{


	unsigned int v = 0;
	unsigned int sign;


	for (int i = 0; i < D1_PACK_BITWIDTH; i++)
	{

		sign = (src[threadIdx.x + i * 24 * 24] > 0);
		v = v | (sign << (D1_PACK_BITWIDTH - 1 - i));

	}

	dst[threadIdx.x] = v;

}



//<<<100, 64>>>
__global__ void bxnorDense1(const unsigned int* wgt, const unsigned int* src, int* dst, const unsigned int m, const unsigned int n) {


	__shared__ int segSum[D1_BLOCK_DIM];

	segSum[threadIdx.x] = 0;
	__syncthreads();




	for (int i = 0; i < D1_PARTS; i++)
		segSum[threadIdx.x] += D1_PACK_BITWIDTH - 2 * (__popc(wgt[blockIdx.x * n + threadIdx.x + i * D1_BLOCK_DIM] ^ src[threadIdx.x + i * D1_BLOCK_DIM]));


	__syncthreads();



	//reduction
#pragma unroll
	for (int i = 1; i < D1_BLOCK_LOG2; i++)
	{
		int range = (D1_BLOCK_DIM >> i);

		if (threadIdx.x < range)
			segSum[threadIdx.x] += segSum[range + threadIdx.x];

		__syncthreads(); //<--- theoratically, this should not be needed here
	}


	if (threadIdx.x < 1)
		dst[blockIdx.x] = segSum[0] + segSum[1];


}





//<<<1,100>>>
__global__ void packRowsDense2(const int *src, unsigned int *dst, const int size)
{
	__shared__ int sh_block[100];

	sh_block[threadIdx.x] = src[threadIdx.x];

	__syncthreads();

	if (threadIdx.x < 4) {

		unsigned int v = 0;
		unsigned int sign;

		//Even with this loop removed (100% shared memory efficiency), performance only improves by 20%
		for (int i = 0; i < PACK_BITWIDTH; i++)
		{
			sign = (sh_block[threadIdx.x * PACK_BITWIDTH + i] > 0);
			v = v | (sign << (PACK_BITWIDTH - 1 - i));

		}

		dst[threadIdx.x] = v;


	}


}






//<<< 1, 100 >>>
__global__ void bxnorDense2(const unsigned int* wgt, const unsigned int* src, int* dst, const unsigned int m, const unsigned int n) {


	int sum = 0;
	for (int i = 0; i < n; i++) //each thread only needs to dot two 4-arrays
		sum += PACK_BITWIDTH - 2 * (__popc(wgt[threadIdx.x*n + i] ^ src[i]));


	dst[threadIdx.x] = sum;

}


//<<< 1, 100>>>
__global__ void packRowsDense3(const int *src, unsigned int *dst, const int size)
{
	__shared__ int sh_block[100];

	sh_block[threadIdx.x] = src[threadIdx.x];

	__syncthreads();

	if (threadIdx.x < 4) {

		unsigned int v = 0;
		unsigned int sign;

		//Even with this loop removed (100% shared memory efficiency), performance only improves by 20%
		for (int i = 0; i < PACK_BITWIDTH; i++)
		{
			sign = (sh_block[threadIdx.x * PACK_BITWIDTH + i] > 0);
			v = v | (sign << (PACK_BITWIDTH - 1 - i));

		}

		dst[threadIdx.x] = v;


	}


}



//<<< 1, (4,4) >>>
__global__ void bxnorDense3(const unsigned int* wgt, const unsigned int* src, int* dst, const unsigned int m, const unsigned int n) {



	int sum = 0;
	for (int i = 0; i < n; i++) {
		sum += PACK_BITWIDTH - 2 * (__popc(wgt[threadIdx.x * n + i] ^ src[i]));

	}
	dst[threadIdx.x] = sum;



}


//Receives 96x96x3 input
//

//<<<(48,1,3),(96,2)>>>
__global__ void Im2Col3d(int *src, int* dst)
{


	src += (blockIdx.x*IM2COL_BLOCK_DIM_Y) * IM2COL_BLOCK_DIM_X + blockIdx.z*(WIDTH  * HEIGHT) - (blockIdx.x>0)*KERNEL_RADIUS*IM2COL_BLOCK_DIM_X;
	dst += blockIdx.x * (IM2COL_BLOCK_DIM_Y * IM2COL_BLOCK_DIM_X) + blockIdx.z*(WIDTH  * HEIGHT * 25) + threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x;  // store as 32x(96*96)


	__shared__  int sh_block[IM2COL_REGION_2D];

	for (int i = 0; i<4; i++)
		if (threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x + i*IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y< IM2COL_REGION_2D)
			sh_block[threadIdx.y * IM2COL_BLOCK_DIM_X + threadIdx.x + i * IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y] = 0;


	__syncthreads();


	//This should probably be a loop
	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y)] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH)];//y 


	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * 2] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH) + IM2COL_BLOCK_DIM_X * 2];//y 

	if (blockIdx.x>0 && blockIdx.x < gridDim.x - 1)
		sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + 4 * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)]
		= src[(threadIdx.x) // x
		+ (threadIdx.y*WIDTH) + 4 * IM2COL_BLOCK_DIM_X];




	__syncthreads();







	unsigned int offset = (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) *threadIdx.y + (threadIdx.x);

	for (int i = 0, k = 0; i < PACK_BITWIDTH; i++)
	{

		//calculate div(i,KERNEL_LENGTH)
		if (i - k * KERNEL_LENGTH == KERNEL_LENGTH)
			k++;

		dst[i*WIDTH*HEIGHT] = sh_block[offset + k * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) + (i - k * KERNEL_LENGTH)];	// store as (25*3)x(96*96)


	}





}





// wgt is 32x75
// src is 75x(96*96)
// dst is 32x(96*96)
// Function of each block:
//		loads the entire weights matrix to shared memory
//		loads a chunk of 3x32, computes 32x32 result, and load next chunk... repeat. So each block computes a 32x(3*32) region in the output. Note: 32x32 (in otutput) is one row of length 32 for each of 32 channels,
//
//<<< 72, (32,32) >>>		
__global__ void xnorConv1(const int* src, int* dst, int* wgt) {



	const int idxLin = threadIdx.y*CONV_BLOCK_DIM_X + threadIdx.x; //local linear index

																   //src += blockIdx.x*(CONV_NUM_CHUNKS*blockDim.x * KERNEL_SIZE*C1 / PACK_BITWIDTH); //each block processes a (32*4)x3 region from src
																   //dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

																   // src is 32x(96*96)
	src += blockIdx.x*(CONV_NUM_CHUNKS * CONV_BLOCK_DIM_X);
	dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

	/*__shared__ unsigned int sh_wgt[FMAPS1 * KERNEL_SIZE1];*/
	__shared__  int sh_im[C1 * 25 * CONV_BLOCK_DIM_X];




	for (int j = 0; j < CONV_NUM_CHUNKS; j++) {

		int sumC = 0;

		//if (threadIdx.y<3)
		sh_im[idxLin] = src[j* CONV_BLOCK_DIM_X + threadIdx.y*WIDTH  * HEIGHT + threadIdx.x];
		sh_im[idxLin + 32 * 32] = src[j* CONV_BLOCK_DIM_X + (32 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];
		if (idxLin + 32 * 32 * 2<32 * 75)
			sh_im[idxLin + 32 * 32 * 2] = src[j* CONV_BLOCK_DIM_X + (32*2 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];

		__syncthreads();




		for (int i = 0; i < 25 * 3; i++) {
			//sumC += PACK_BITWIDTH - 2 * (__popc(wgtConv1[threadIdx.y *C1 + i] ^ sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i]));
			sumC += wgt[threadIdx.y * 25 * 3 + i] * sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i];



		}



		dst[threadIdx.y * WIDTH * HEIGHT + CONV_BLOCK_DIM_X * j + threadIdx.x] = (sumC);

		__syncthreads();

	}


}

//Receives 96x96x3 input
//

//<<<(48,1,3),(96,2)>>>
__global__ void fIm2Col3d(float *src, float* dst)
{

	
	src += (blockIdx.x*IM2COL_BLOCK_DIM_Y) * IM2COL_BLOCK_DIM_X + blockIdx.z*(WIDTH  * HEIGHT) - (blockIdx.x>0)*KERNEL_RADIUS*IM2COL_BLOCK_DIM_X;
	dst += blockIdx.x * (IM2COL_BLOCK_DIM_Y * IM2COL_BLOCK_DIM_X) + blockIdx.z*(WIDTH  * HEIGHT * 25) + threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x;  // store as 32x(96*96)


	__shared__  float sh_block[IM2COL_REGION_2D];

	for (int i = 0; i<4; i++)
		if (threadIdx.y*IM2COL_BLOCK_DIM_X + threadIdx.x + i*IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y< IM2COL_REGION_2D)
			sh_block[threadIdx.y * IM2COL_BLOCK_DIM_X + threadIdx.x + i * IM2COL_BLOCK_DIM_X * IM2COL_BLOCK_DIM_Y] = 0;


	__syncthreads();


	//This should probably be a loop
	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y)] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH)];//y 


	sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * 2] //y

		= src[(threadIdx.x)  // x
		+ (threadIdx.y*WIDTH) + IM2COL_BLOCK_DIM_X * 2];//y 

	if (blockIdx.x>0 && blockIdx.x < gridDim.x - 1)
		sh_block[KERNEL_RADIUS + (blockIdx.x == 0)*(IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)*KERNEL_RADIUS // offset
		+ (threadIdx.x)  // x
		+ ((IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) * threadIdx.y) + 4 * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS)]
		= src[(threadIdx.x) // x
		+ (threadIdx.y*WIDTH) + 4 * IM2COL_BLOCK_DIM_X];




	__syncthreads();







	unsigned int offset = (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) *threadIdx.y + (threadIdx.x);

	for (int i = 0, k = 0; i < PACK_BITWIDTH; i++)
	{

		//calculate div(i,KERNEL_LENGTH)
		if (i - k * KERNEL_LENGTH == KERNEL_LENGTH)
			k++;

		dst[i*WIDTH*HEIGHT] = sh_block[offset + k * (IM2COL_BLOCK_DIM_X + 2 * KERNEL_RADIUS) + (i - k * KERNEL_LENGTH)];	// store as (25*3)x(96*96)


	}





}
__global__ void fxnorConv1(const float*  src, float* dst, float*  wgt) {



	const  int idxLin = threadIdx.y*CONV_BLOCK_DIM_X + threadIdx.x; //local linear index

																   //src += blockIdx.x*(CONV_NUM_CHUNKS*blockDim.x * KERNEL_SIZE*C1 / PACK_BITWIDTH); //each block processes a (32*4)x3 region from src
																   //dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

																   // src is 32x(96*96)
	src += blockIdx.x*(CONV_NUM_CHUNKS * CONV_BLOCK_DIM_X);
	dst += blockIdx.x * CONV_NUM_CHUNKS*CONV_BLOCK_DIM_X;

//	__shared__ float sh_wgt[FMAPS1 * 75];
	__shared__  float sh_im[C1 * 25 * CONV_BLOCK_DIM_X];




	for (int j = 0; j < CONV_NUM_CHUNKS; j++) {

		float sumC = 0;

		//if (threadIdx.y<3)
		sh_im[idxLin] = src[j* CONV_BLOCK_DIM_X + threadIdx.y*WIDTH  * HEIGHT + threadIdx.x];
		sh_im[idxLin + 32 * 32] = src[j* CONV_BLOCK_DIM_X + (32 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];

		if (threadIdx.y < 11)
			sh_im[idxLin + 32 * 32 * 2] = src[j* CONV_BLOCK_DIM_X + (32 * 2 + threadIdx.y)*WIDTH  * HEIGHT + threadIdx.x];
	

		__syncthreads();




		for (int i = 0; i < 25 * 3; i++) {
			//sumC += PACK_BITWIDTH - 2 * (__popc(wgtConv1[threadIdx.y *C1 + i] ^ sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i]));
			sumC += wgt[threadIdx.y * 25 * 3 + i] * sh_im[threadIdx.x + CONV_BLOCK_DIM_X* i];
		//	sumC += wgt[threadIdx.y * 25 * 3 + i] * src[j* CONV_BLOCK_DIM_X +  threadIdx.x + WIDTH  * HEIGHT *i];



		}



		dst[threadIdx.y * WIDTH * HEIGHT + CONV_BLOCK_DIM_X * j + threadIdx.x] = (sumC);

		__syncthreads();

	}


}




#endif

//This function is from the Nvidia CUDA samples
template <typename T>
void matrixMulCPU(T *C, const T *A, const T *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j)
		{
			T sum = 0;

			for (unsigned int k = 0; k < wA; ++k)
			{
				T a = A[i * wA + k];
				T b = B[k * wB + j];
				sum += a * b;
			}

			C[i * wB + j] = (T)sum;
		}
}

//
//template <typename T>
//void CompareResults(T* gold, T* cudaOutput, size_t n) {
//
//	for (int i = 0; i < n; i++)
//		if (gold[i] != cudaOutput[i])
//		{
//			printf("***DIFFERENCES FOUND***; first at %i (%i != %i).\n", i, gold[i], cudaOutput[i]);
//
//			return;
//		}
//	printf("PASSED\n");
//
//}
//
//
//void CompareResults(unsigned int* gold, unsigned int* cudaOutput, size_t n) {
//
//	for (int i = 0; i < n; i++)
//		if (gold[i] != cudaOutput[i])
//		{
//			printf("***DIFFERENCES FOUND***; first at %u: \n", i);
//
//			printf("gold:   ");
//			for (int j = 0; j < 40; j++) {
//
//				if (j + i - 10 == i)
//					printf("[%u] ", gold[j + i - 10]);
//				else
//					printf("%u ", gold[j + i - 10]);
//			}
//			printf("\noutput: ");
//			for (int j = 0; j < 40; j++) {
//
//				if (j + i - 10 == i)
//					printf("[%u] ", cudaOutput[j + i - 10]);
//				else
//					printf("%u ", cudaOutput[j + i - 10]);
//			}
//
//			return;
//		}
//	printf("PASSED\n");
//
//}

void normalizeImage(int* dst, int* src, int a, int b) {

	int maxIm = *src;
	int minIm = *src;


	for (int i = 0; i < 3 * WIDTH*HEIGHT; i++) {

		int x = *src++;
		if (x > maxIm)
			maxIm = x;
		if (x < minIm)
			minIm = x;


	}
	src -= 3 * WIDTH*HEIGHT;

	//It's not necessary to cast to double (outputs will be slightly different, but final results remain correct)
	for (int i = 0; i < 3 * WIDTH*HEIGHT; i++)
		dst[i] = double((b - a)*(src[i] - minIm)) / double((maxIm - minIm) )+ a >0?1:-1;




}

void CompareResults(float* gold, float* cudaOutput, size_t n) {

	for (int i = 0; i < n; i++)
		if (round(gold[i]) != round(cudaOutput[i]))
		{
			printf("***DIFFERENCES FOUND***; first at %u (gold(%f) != %f).\n", i, gold[i], cudaOutput[i]);

			return;
		}
	printf("PASSED\n");

}

void CompareResults(int* gold, int* cudaOutput, size_t n) {

	for (int i = 0; i < n; i++)
		if (round(gold[i]) != round(cudaOutput[i]))
		{
			printf("***DIFFERENCES FOUND***; first at %u (%i != %i).\n", i, gold[i], cudaOutput[i]);

			return;
		}
	printf("PASSED\n");

}

void CompareResults(unsigned int* gold, unsigned int* cudaOutput, size_t n) {

	for (int i = 0; i < n; i++)
		if (round(gold[i]) != round(cudaOutput[i]))
		{
			printf("***DIFFERENCES FOUND***; first at %u (%u != %u).\n", i, gold[i], cudaOutput[i]);

			return;
		}
	printf("PASSED\n");

}

double L1(float* gold, float* output, size_t n) {
	double norm = 0.0;

	for (int i = 0; i < n; i++)
		norm += gold[i] - output[i];

	return norm;
}

const char* GetAlgStr(cudnnConvolutionFwdAlgo_t alg) {
	switch (alg) {
	case     CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
		return"Implicit GEMM";
	case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
		return"Implicit GEMM with auxiliary memory";
	case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
		return"GEMM";
	case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
		return"Direct Convolution";
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
		return"Fast Fourier Transform";
	case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
		return"Fast Fourier Transform with tiling";
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
		return"Winograd Transform";
	case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
		return"Winograd Transform (non-fused)";
	default:
		return "INVALID ALGORITHM ARGUMENT";

	}




}

int main()
{
	/*
	int i1 = 3;
	int i2 = 4;
	printf("Xnor: %i\n",~(i1 ^ i2));*/

	


	cudaError_t cudaStatus;
	cudaDeviceProp deviceProp;
	int device= 0;
	CUDA_ERROR(cudaSetDevice(device), "cudaSetDevice");
	CUDA_ERROR(cudaGetDevice(&device), "cudaGetDevice");
	CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, device),"cudaGetDeviceProperties");
	printf("Using GPU Device [%d]: %s \nCompute capability\t %d.%d\nMultiprocessors\t\t %i\nTotal Global Memory\t %li MB\n\n", 
		device, deviceProp.name, deviceProp.major, deviceProp.minor,deviceProp.multiProcessorCount,deviceProp.totalGlobalMem/(1024*1024));

	
	//Fill memory so that out-of-bound accesses don't survive
	float* filler = (float*)malloc(100 * 100 * 100 * sizeof(float));
	float* dev_filler;
	CUDA_ERROR(cudaMalloc((void**)&dev_filler, 1000 * 100 * 100 * sizeof(float)), "cudaMalloc");



	//profiling
	float ms_t = 0.0f;
	cudaEvent_t start, stop;

	//__host__ buffers
	int*		  im ;
	float*		  fim;
	int*		  normalizedIm;
	int* batch = (int*)malloc(FRAMES*WIDTH*HEIGHT*C1 * sizeof(int));
	for (int i = 0; i < FRAMES*WIDTH*HEIGHT*C1; i++)
		batch[i] = i;

	float* fbatch = (float*)malloc(FRAMES*WIDTH*HEIGHT*C1 * sizeof(float));
	for (int i = 0; i < FRAMES*WIDTH*HEIGHT*C1; i++)
		fbatch[i] = float(i);

	CUDA_ERROR(cudaMallocHost((void**)&im, WIDTH*HEIGHT * C1 * sizeof(int)), "cudaMallocHost");
	CUDA_ERROR(cudaMallocHost((void**)&fim, WIDTH*HEIGHT * C1 * sizeof(float)), "cudaMallocHost");

	int* Im2Col = (int*)malloc((KERNEL_SIZE*C1)*(WIDTH *HEIGHT) * sizeof(int));
	unsigned int* pkIm2Col = (unsigned int*)malloc((C1)*(WIDTH *HEIGHT) * sizeof(unsigned int));

	//weights (packed)
	unsigned int* pkwConv1 = (unsigned int*)malloc(FMAPS1 * KERNEL_SIZE*C1 / PACK_BITWIDTH * sizeof(unsigned int));
	unsigned int* pkwConv2 = (unsigned int*)malloc(FMAPS1 * KERNEL_SIZE*C2 / PACK_BITWIDTH * sizeof(unsigned int));
	unsigned int* pkwD1 = (unsigned int*)malloc(576 * 100 * sizeof(unsigned int));
	unsigned int* pkwD2 = (unsigned int*)malloc(4 * 100 * sizeof(unsigned int));
	unsigned int* pkwD3 = (unsigned int*)malloc(4 * 4 * sizeof(unsigned int));

	//_fp
	float* fIm2Col = (float*)malloc((KERNEL_SIZE*C1)*(WIDTH *HEIGHT) * sizeof(float));
	float* fwConv1 = (float*)malloc(FMAPS1*KERNEL_SIZE*C1 * sizeof(float));
	float* fwConv2 = (float*)malloc(FMAPS1*KERNEL_SIZE*C2 * sizeof(float));
	float* fwD1 = (float*)malloc(24*24*32*100 * sizeof(float));;
	float* fwD2 = (float*)malloc(100*100 * sizeof(float));;
	float* fwD3 = (float*)malloc(100*4 * sizeof(float));;

	//for reading computation resuls (only used for debugging and verification)
	int*		  conv1 = (int*)malloc(WIDTH*HEIGHT*FMAPS1 * sizeof(int));
	int*		  mxpConv1 = (int*)malloc(WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(int));
	unsigned int* pkMxpConv1 = (unsigned int*)malloc(WIDTH / 2 * HEIGHT / 2 * KERNEL_SIZE*C2 / PACK_BITWIDTH * sizeof(unsigned int));
	int*		  conv2 = (int*)malloc(WIDTH*HEIGHT * FMAPS1 * sizeof(int));
	int*		  mxpConv2 = (int*)malloc(WIDTH / 4 * HEIGHT / 4 * FMAPS1 * sizeof(int));
	unsigned int* pkH = (unsigned int*)malloc(24 * 24 *800 * sizeof(unsigned int)); 
	int*		  d1 = (int*)malloc(100 * sizeof(int));
	int*		  d2 = (int*)malloc(100 * sizeof(int));
	int*		  d3 = (int*)malloc(4 * sizeof(int));
	unsigned int* pkD1 = (unsigned int*)malloc(4 * sizeof(unsigned int));
	unsigned int* pkD2 = (unsigned int*)malloc(4 * sizeof(unsigned int));

	//_fp
	float*		  fconv1 = (float*)malloc(WIDTH*HEIGHT*FMAPS1 * sizeof(float));
	float*		  fmxpConv1 = (float*)malloc(WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(float));
	float*		  fconv2 = (float*)malloc(WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(float));
	float*		  fmxpConv2 = (float*)malloc(WIDTH / 4 * HEIGHT / 4 * FMAPS1 * sizeof(float));
	float*		  fd1 = (float*)malloc(100 * sizeof(float));
	float*		  fd2 = (float*)malloc(100 * sizeof(float));
	float*		  fd3 = (float*)malloc(4 * sizeof(float));

	//computation results verification

	unsigned int* gold_pkIm2Col = (unsigned int*)malloc(3 * 96 * 96 * sizeof(unsigned int));

	 int* gold_Im2Col = ( int*)malloc(3*25 * 96 * 96 * sizeof( int));
	 float* gold_fIm2Col = (float*)malloc(3 * 25 * 96 * 96 * sizeof(float));

	int*		  gold_mxpConv1 = (int*)malloc(48 * 48 * 32 * sizeof(int));
	unsigned int* gold_pkMxpConv1 = (unsigned int*)malloc(48 * 48 * KERNEL_SIZE*C2 / PACK_BITWIDTH * sizeof(unsigned int));
	int*		  gold_mxpConv2 = (int*)malloc(24 * 24 * 32 * sizeof(int));
	unsigned int* gold_pkMxpConv2 = (unsigned int*)malloc(24 * 24 * 25 * sizeof(unsigned int));
	int*		  gold_conv1 = (int*)malloc(96 * 96 * 32 * sizeof(int));

	int*		  gold_conv2 = (int*)malloc(48 * 48 * 32 * sizeof(int));


	unsigned int* gold_pkH = (unsigned int*)malloc(576 * 1 * sizeof(unsigned int));
	int* gold_d1 = (int*)malloc(100 * sizeof(int));
	int* gold_d2 = (int*)malloc(100 * sizeof(int));
	int* gold_d3 = (int*)malloc(4 * sizeof(int));

	//_fp
	float* gold_fconv1 = (float*)malloc(96 * 96 * 32 * sizeof(float));
	float* gold_fconv2 = (float*)malloc(48 * 48 * 32 * sizeof(float));
	float* gold_fh = (float *)malloc(24*24*32 * sizeof(float ));
	float* gold_fd1 = (float*)malloc(100 * sizeof(float));
	float* gold_fd2 = (float*)malloc(100 * sizeof(float));
	float* gold_fd3 = (float*)malloc(4 * sizeof(float));

	//__device__ buffers
	int* dev_im;	 
	int* dev_Im2Col;
	unsigned int* dev_pkIm2Col;
	int*		  dev_conv1;
	int*		  dev_mxpConv1;
	int*		  dev_conv2;
	int*		  dev_mxpConv2;
	unsigned int* dev_pkMxpConv1;
	unsigned int* dev_pkH;

	int* 		  dev_d1;
	int* 		  dev_d2;
	int* 		  dev_d3;
	unsigned int* dev_pkD1;
	unsigned int* dev_pkD2;

	//weights
	 int* dev_wConv1;
	 unsigned int* dev_pkwConv1;
	 unsigned int* dev_pkwConv2;
 	 unsigned int* dev_pkwD1;
	 unsigned int* dev_pkwD2;
	 unsigned int* dev_pkwD3;


	 //_fp
	 float*		  dev_fim;
	 float*		  dev_fIm2Col;
	 float*		  dev_fconv1;
	 float*		  dev_fmxpConv1;
	 float*		  dev_fconv2;
	 float*		  dev_fmxpConv2;
	 float* 		  dev_fh;
	 float* 		  dev_fd1;
	 float* 		  dev_fd2;
	 float* 		  dev_fd3;
	 float* dev_ftwD1;
	 //weights
	 float* dev_fwConv1;
	 float* dev_fwConv2;
	 float* dev_fwD1;
	 float* dev_fwD2;
	 float* dev_fwD3;






	 uclLoadDatai("im.bin", im, 3 * 96 * 96);
	 uclLoadData("fim.bin", fim, 3 * 96 * 96);

	// load weights (all packed)
	uclLoadDataui("wgt/pkwConv1.bin", pkwConv1, (32 * 3 * 25) / PACK_BITWIDTH);
	uclLoadDataui("wgt/pkwConv2.bin", pkwConv2, (32 * 25 * 32) / PACK_BITWIDTH);
	uclLoadDataui("wgt/pkwD1.bin", pkwD1, 24 * 24 * 100);
	uclLoadDataui("wgt/pkwD2.bin", pkwD2, 4 * 100);
	uclLoadDataui("wgt/pkwD3.bin", pkwD3, 4 * 4);

	//_fp
	uclLoadData("wgtFP/fwConv1.bin", fwConv1, (32 * 3 * 25));
	uclLoadData("wgtFP/fwConv2.bin", fwConv2, (32 * 32 * 25));
	uclLoadData("wgtFP/fwD1.bin", fwD1, 24 * 24 * 32* 100);
	uclLoadData("wgtFP/fwD2.bin", fwD2, 100 * 100);
	uclLoadData("wgtFP/fwD3.bin", fwD3, 100 * 4);



	//load gold
	//uclLoadDataui("gold/gold_pkImCol.bin", gold_pkIm2Col, 3 * 96 * 96);
	//uclLoadDatai("gold/gold_conv1.bin", gold_conv1, 32 * 96 * 96);
	//uclLoadDatai("gold/gold_mxpConv1.bin", gold_mxpConv1, 32 * 48 * 48);
	//uclLoadDataui("gold/gold_pkMxpConv1.bin", gold_pkMxpConv1, (48 * 48) * KERNEL_SIZE*C2 / PACK_BITWIDTH); //'im2col'
	//uclLoadDatai("gold/gold_conv2.bin", gold_conv2, 32 * 48 * 48);
	//uclLoadDatai("gold/gold_mxpConv2.bin", gold_mxpConv2, 32 * 24 * 24);
	//uclLoadDataui("gold/gold_pkH.bin", gold_pkH, 24 * 24);
	//uclLoadDatai("gold/gold_d1.bin", gold_d1, 100);
	//uclLoadDatai("gold/gold_d2.bin", gold_d2, 100);
	//uclLoadDatai("gold/gold_d3.bin", gold_d3, 4);


	//_fp
	//uclLoadData("goldFP/gold_fImCol.bin", gold_fIm2Col, 3 * 25 * 96 * 96);
	//uclLoadData("goldFP/gold_fconv1.bin", gold_fconv1, 32 * 96 * 96);
	//uclLoadData("gold/gold_fmxpConv1.bin", gold_fmxpConv1, 32 * 48 * 48);

	//uclLoadData("goldFP/gold_fconv2.bin", gold_fconv2, 32 * 48 * 48);
	//uclLoadData("gold/gold_fmxpConv2.bin", gold_mxpConv2, 32 * 24 * 24);
	//uclLoadData("goldFP/gold_fh.bin", gold_fh, 24 * 24*32);
	//uclLoadData("gold/gold_fd1.bin", gold_fd1, 100);
	//uclLoadData("gold/gold_fd2.bin", gold_fd2, 100);
	//uclLoadData("gold/gold_fd3.bin", gold_fd3, 4);
	

	//Allocate device memory + pinned memory

	CUDA_ERROR(cudaMalloc((void**)&dev_im, WIDTH * HEIGHT * 3 * sizeof(int)), "cudaMalloc");

	CUDA_ERROR(cudaMalloc((void**)&dev_Im2Col, (3 * 25)*(WIDTH * HEIGHT) * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkIm2Col, (3)*(WIDTH * HEIGHT) * sizeof(unsigned int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_conv1, WIDTH * HEIGHT * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_conv2, WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_mxpConv1, WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkMxpConv1, WIDTH / 2 * HEIGHT / 2 * C2 * sizeof(unsigned int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_mxpConv2, WIDTH / 4 * HEIGHT / 4 * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_d1, 100 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_d2, 100 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_d3, 4 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkD1, 4 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkD2, 4 * sizeof(int)), "cudaMalloc");


	CUDA_ERROR(cudaMalloc((void**)&dev_wConv1, FMAPS1 * KERNEL_SIZE*C1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkwConv1, FMAPS1 * KERNEL_SIZE*C1 / PACK_BITWIDTH * sizeof(unsigned int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkwConv2, FMAPS1 * KERNEL_SIZE*C2 / PACK_BITWIDTH * sizeof(unsigned int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkwD1, WIDTH / 4 * HEIGHT / 4 * 100 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkwD2, 5 * 5 * 3 * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkwD3, 5 * 5 * 32 * FMAPS1 * sizeof(int)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_pkH, WIDTH / 4 * HEIGHT / 4 * 1 * sizeof(unsigned int)), "cudaMalloc");

	//_fp

	CUDA_ERROR(cudaMalloc((void**)&dev_fim, WIDTH * HEIGHT * 3 * sizeof(float)), "cudaMalloc");

	CUDA_ERROR(cudaMalloc((void**)&dev_fIm2Col, (3 * 25)*(WIDTH * HEIGHT) * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fconv1, WIDTH * HEIGHT * FMAPS1 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fmxpConv1, WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fconv2, WIDTH/2 * HEIGHT/2 * FMAPS1 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fmxpConv2, WIDTH / 2 * HEIGHT / 2 * FMAPS1 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fd1, 100 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fd2, 100 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fd3, 4 * sizeof(float)), "cudaMalloc");

	

	CUDA_ERROR(cudaMalloc((void**)&dev_fwConv1, FMAPS1 * KERNEL_SIZE*C1 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fwConv2, FMAPS1 * KERNEL_SIZE*C2 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fwD1, 32*WIDTH / 4 * HEIGHT / 4 * 100 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_ftwD1, 32 * WIDTH / 4 * HEIGHT / 4 * 100 * sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fwD2, 100*100* sizeof(float)), "cudaMalloc");
	CUDA_ERROR(cudaMalloc((void**)&dev_fwD3, 100*4 * FMAPS1 * sizeof(float)), "cudaMalloc");


	//Copy weights to device
	CUDA_ERROR(cudaMemcpy(dev_pkwConv1, pkwConv1, FMAPS1 * KERNEL_SIZE*C1 / PACK_BITWIDTH * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_pkwConv2, pkwConv2, FMAPS1 * KERNEL_SIZE*C2 / PACK_BITWIDTH * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_pkwD1, pkwD1, WIDTH / 4 * HEIGHT / 4 * 100 * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_pkwD2, pkwD2, 4 * 100 * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_pkwD3, pkwD3, 4 * 4 * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");

	//_fp
	CUDA_ERROR(cudaMemcpy(dev_fwConv1, fwConv1, FMAPS1 * KERNEL_SIZE*C1 * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_fwConv2, fwConv2, FMAPS1 * KERNEL_SIZE*C2 * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_fwD1, fwD1, 32*WIDTH / 4 * HEIGHT / 4 * 100 * sizeof(float ), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_fwD2, fwD2, 100 * 100 * sizeof(float ), cudaMemcpyHostToDevice), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(dev_fwD3, fwD3, 4*100 * sizeof(float ), cudaMemcpyHostToDevice), "cudaMemcpy");

	//Copy to constant memory (not used)
	CUDA_ERROR(cudaMemcpyToSymbol(wgtConv1, pkwConv1, FMAPS1 * KERNEL_SIZE*C1 / PACK_BITWIDTH * sizeof(unsigned int)), "cudaMemcpyToSymbol");
	//CUDA_ERROR(cudaMemcpyToSymbol(wgtConv1, dwConv1, FMAPS1 * KERNEL_SIZE*C1 * sizeof(int)), "cudaMemcpyToSymbol");


	//----------------------------------------------
	//      cuDNN Setup
	//----------------------------------------------

	cudnnHandle_t hnd_cuDNN;
	cudnnStatus_t status;
	cudnnTensorDescriptor_t desc_tsrIm;
	cudnnTensorDescriptor_t desc_tsrConv1;
	cudnnTensorDescriptor_t desc_tsrConv2;
	cudnnTensorDescriptor_t desc_tsrMxp;
	cudnnTensorDescriptor_t desc_tsrMxp2;
	cudnnTensorDescriptor_t desc_tsrD1;
	cudnnTensorDescriptor_t desc_tsrD2;


	cudnnFilterDescriptor_t desc_wgtConv1;
	cudnnFilterDescriptor_t desc_wgtConv2;
	cudnnConvolutionDescriptor_t desc_convolution1;
	cudnnConvolutionDescriptor_t desc_convolution2;
	cudnnPoolingDescriptor_t desc_MaxPooling1;
	cudnnPoolingDescriptor_t desc_MaxPooling2;
	cudnnConvolutionFwdAlgo_t alg_convolution1;
	cudnnConvolutionFwdAlgo_t alg_convolution2;
	cudnnActivationDescriptor_t desc_relu1;

	float*		  dev_workspace;
	float*		  dev_workspace2;
	size_t workspace_bytes = 0;
	size_t workspace_bytes2 = 0;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	float total = 0.0;

	printf("\ncuDNN Initialization: %s\n", cudnnGetErrorString(cudnnCreate(&hnd_cuDNN)));

	//Initialize descriptors
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrIm), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrConv1), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrMxp), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrConv2), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrMxp2), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrD1), "cudnnCreateTensorDescriptor");
	CUDNN_ERROR(cudnnCreateTensorDescriptor(&desc_tsrD2), "cudnnCreateTensorDescriptor");

	CUDNN_ERROR(cudnnCreateFilterDescriptor(&desc_wgtConv1), "cudnnFilterDescriptor_t");
	CUDNN_ERROR(cudnnCreateFilterDescriptor(&desc_wgtConv2), "cudnnFilterDescriptor_t");

	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&desc_convolution1), "cudnnCreateConvolutionDescriptor");
	CUDNN_ERROR(cudnnCreatePoolingDescriptor(&desc_MaxPooling1), "cudnnCreatePoolingDescriptor");
	CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&desc_convolution2), "cudnnCreateConvolutionDescriptor");
	CUDNN_ERROR(cudnnCreatePoolingDescriptor(&desc_MaxPooling2), "cudnnCreatePoolingDescriptor");
	CUDNN_ERROR(cudnnCreateActivationDescriptor(&desc_relu1), "cudnnCreateActivationDescriptor");

	
	//Set input/results tensors
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrIm, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, HEIGHT, WIDTH), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrConv1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 32, HEIGHT, WIDTH), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrMxp, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 32, HEIGHT / 2, WIDTH / 2), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrConv2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 32, HEIGHT/2, WIDTH/2), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrMxp2, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 32, HEIGHT / 4, WIDTH / 4), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrD1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 100, 1), "cudnnSetTensor4dDescriptor");
	CUDNN_ERROR(cudnnSetTensor4dDescriptor(desc_tsrD2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 100, 1), "cudnnSetTensor4dDescriptor");
	

	//Weights
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(desc_wgtConv1, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 3, 5, 5), "cudnnSetFilter4dDescriptor");
	CUDNN_ERROR(cudnnSetFilter4dDescriptor(desc_wgtConv2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 32, 5, 5), "cudnnSetFilter4dDescriptor");

	//Layer descs
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(desc_convolution1, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION), "cudnnSetConvolution2dDescriptor");
	CUDNN_ERROR(cudnnSetPooling2dDescriptor(desc_MaxPooling1, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2), "cudnnSetPooling2dDescriptor");
	CUDNN_ERROR(cudnnSetConvolution2dDescriptor(desc_convolution2, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION), "cudnnSetConvolution2dDescriptor");
	CUDNN_ERROR(cudnnSetPooling2dDescriptor(desc_MaxPooling2, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2), "cudnnSetPooling2dDescriptor");
	CUDNN_ERROR(cudnnSetActivationDescriptor(desc_relu1, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0), "cudnnSetActivationDescriptor");

	//Choose convolution algorithm
	CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(hnd_cuDNN, desc_tsrIm, desc_wgtConv1, desc_convolution1, desc_tsrConv1, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &alg_convolution1), "cudnnGetConvolutionForwardAlgorithm");
	CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(hnd_cuDNN, desc_tsrMxp, desc_wgtConv2, desc_convolution2, desc_tsrConv2, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &alg_convolution2), "cudnnGetConvolutionForwardAlgorithm");

	alg_convolution1 = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	alg_convolution2 = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

	CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(hnd_cuDNN,
		desc_tsrIm,
		desc_wgtConv1,
		desc_convolution1,
		desc_tsrConv1,
		alg_convolution1,
		&workspace_bytes), "cudnnGetConvolutionForwardWorkspaceSize");

	CUDA_ERROR(cudaMalloc((void**)&dev_workspace, workspace_bytes), "cudaMalloc");

	CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(hnd_cuDNN,
		desc_tsrMxp,
		desc_wgtConv2,
		desc_convolution2,
		desc_tsrConv2,
		alg_convolution2,
		&workspace_bytes2), "cudnnGetConvolutionForwardWorkspaceSize");

	CUDA_ERROR(cudaMalloc((void**)&dev_workspace2, workspace_bytes2), "cudaMalloc");

	printf("cuDNN convolution algorithm for 1st layer: %s\n", GetAlgStr(alg_convolution1));
	printf("cuDNN Convolution algorithm for 2nd layer: %s\n", GetAlgStr(alg_convolution2));







	//----------------------------------------------
	//      cuBLAS Setup
	//----------------------------------------------

	cublasHandle_t hnd_cuBLAS;

	printf("\ncuBLAS Initialization: %s\n", _cudaGetErrorEnum(cublasCreate(&hnd_cuBLAS)));




	//Define workspace dimensions for each kernel


	dim3 blocks_Im2Col(96 / IM2COL_BLOCK_DIM_Y, 1, 3);
	dim3 threads_Im2Col(IM2COL_BLOCK_DIM_X, IM2COL_BLOCK_DIM_Y, 1);
	dim3 blocks_Conv1(96 * 96 / (CONV_BLOCK_DIM_X*CONV_NUM_CHUNKS), 1, 1);
	dim3 threads_Conv1(CONV_BLOCK_DIM_X, CONV_BLOCK_DIM_Y, 1);

	dim3 blocks_MaxPool1(3, 3, 32);
	dim3 threads_MaxPool1(32, 32, 1);

	dim3 blocks_Im2Col2(48 / IM2COL_2_BLOCK_DIM_Y, 1, 32);
	dim3 threads_Im2Col2(IM2COL_2_BLOCK_DIM_X, IM2COL_2_BLOCK_DIM_Y, 1);

	dim3 blocks_Conv2(48 * 48 / (CONV_BLOCK_DIM_X*CONV2_NUM_CHUNKS), 1, 1);
	dim3 threads_Conv2(CONV_BLOCK_DIM_X, CONV_BLOCK_DIM_Y, 1);

	dim3 blocks_MaxPool2(3, 3, 32);
	dim3 threads_MaxPool2(16, 16, 1);


	dim3 blocks_pkRowsDense1(1, 1, 1);
	dim3 threads_pkRowsDense1(24 * 24, 1, 1);

	dim3 blocks_Dense1(100, 1);
	dim3 threads_Dense1(D1_BLOCK_DIM, 1);


	dim3 blocks_pkRowsDense2(1, 1, 1);
	dim3 threads_pkRowsDense2(100, 1, 1);

	dim3 blocks_Dense2(1, 1, 1);
	dim3 threads_Dense2(100, 1, 1);


	dim3 blocks_pkRowsDense3(1, 1, 1);
	dim3 threads_pkRowsDense3(100, 1, 1);

	dim3 blocks_Dense3(1, 1, 1);
	dim3 threads_Dense3(4, 4, 1);

	


	cudaStatus = cudaGetLastError();
	

	if (cudaStatus == cudaSuccess) {




		CUDA_ERROR(cudaEventCreate(&start),"cudaEventCreate");
		CUDA_ERROR(cudaEventCreate(&stop),"cudaEventCreate");



		printf("-----------------------------------\nTesting with %i frames, repeat %i times.\n",FRAMES,REPEAT-1);

#ifdef MEASURE_MEMCPY	

		printf("MEMCPY included in measurements.\n");
#else
		printf("MEMCPY NOT included in measurements (testing average runtime of kernels only).\n");

#endif


#ifdef MEASURE_MEMCPY
		cudaDeviceSynchronize();
		cudaEventRecord(start, NULL);
		cudaDeviceSynchronize();
		//normalize image
		cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);


		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {



				fIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_fim, dev_fIm2Col);

#if FRAMES>1

				cudaDeviceSynchronize();
				fxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_fIm2Col, dev_fconv1, dev_fwConv1);
				cudaMemcpy(dev_fim, &fbatch[i*WIDTH*HEIGHT*C1], WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
#else

				fxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_fIm2Col, dev_fconv1, dev_fwConv1);
#endif

			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));

#else

		cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaEventRecord(start, NULL);

		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {


				fIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_fim, dev_fIm2Col);
				fxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_fIm2Col, dev_fconv1, dev_fwConv1);


			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));


#endif

		printf("\nFull precision convolution (1st Layer only) %f us (%1.1f ~fps)\n", 1000 * ms_t / (FRAMES*REPEAT), REPEAT*FRAMES / (ms_t / 1000.0));


		//CUDA_ERROR(cudaMemcpy(fIm2Col, dev_fIm2Col, WIDTH * HEIGHT *(3 * 25) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
		//CUDA_ERROR(cudaMemcpy(fconv1, dev_fconv1, 32 * 96 * 96 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");

		//printf("\nIm2Col3d comparison: ");
		//CompareResults(gold_fIm2Col, fIm2Col, WIDTH*HEIGHT * 3 * 25);
		//printf("Conv1 comparison: ");
		//CompareResults(gold_fconv1, fconv1, 96 * 96 * 32);








#ifdef MEASURE_MEMCPY
		cudaDeviceSynchronize();
		cudaEventRecord(start, NULL);
		cudaDeviceSynchronize();
		//normalize image
		cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);


		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {



				fIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_fim, dev_fIm2Col);

#if FRAMES>1

				cudaDeviceSynchronize();
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 96 * 96, 32, 75, &alpha, dev_fIm2Col, 96 * 96, dev_fwConv1, 75, &beta, dev_fconv1, 96 * 96);
				cudaMemcpy(dev_fim, &fbatch[i*WIDTH*HEIGHT*C1], WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
#else

				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 96 * 96, 32, 75, &alpha, dev_fIm2Col, 96 * 96, dev_fwConv1, 75, &beta, dev_fconv1, 96 * 96);
#endif

			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));

#else

		cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaEventRecord(start, NULL);

		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {


				fIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_fim, dev_fIm2Col);
				cublasSgemm(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N, 96 * 96, 32, 75, &alpha, dev_fIm2Col, 96 * 96, dev_fwConv1, 75, &beta, dev_fconv1, 96 * 96);


			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));


#endif

		printf("\nFP convolution + cuBLAS (1st Layer only) %f us (%1.1f ~fps)\n", 1000 * ms_t / (FRAMES*REPEAT), REPEAT*FRAMES / (ms_t / 1000.0));

	//	CUDA_ERROR(cudaMemcpy(fIm2Col, dev_fIm2Col, WIDTH * HEIGHT *(3 * 25) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//	CUDA_ERROR(cudaMemcpy(fconv1, dev_fconv1, 32 * 96 * 96 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");

//		printf("\nIm2Col3d comparison: ");
//		CompareResults(gold_fIm2Col, fIm2Col, WIDTH*HEIGHT * 3 * 25);
//		printf("L1: %f\n", L1(gold_fIm2Col, fIm2Col, 96 * 96 * 32));
//		printf("Conv1 comparison: ");
//		CompareResults(gold_fconv1, fconv1, 96 * 96 * 32);
//		printf("L1: %f\n", L1(gold_fconv1, fconv1, 96 * 96 * 32));

//cuDNN + cuBLAS
#ifdef MEASURE_MEMCPY
		cudaDeviceSynchronize();
		cudaEventRecord(start, NULL);
		cudaDeviceSynchronize();
		//normalize image
		cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);


		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {



				fIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_fim, dev_fIm2Col);

#if FRAMES>1

				cudaDeviceSynchronize();
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 96 * 96, 32, 75, &alpha, dev_fIm2Col, 96 * 96, dev_fwConv1, 75, &beta, dev_fconv1, 96 * 96);
				cudaMemcpy(dev_fim, &fbatch[i*WIDTH*HEIGHT*C1], WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
#else

				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 96 * 96, 32, 75, &alpha, dev_fIm2Col, 96 * 96, dev_fwConv1, 75, &beta, dev_fconv1, 96 * 96);
#endif

			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));

#else

	

		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {

	//cudaMemcpy(dev_fim, fim, WIDTH * HEIGHT * C1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fim, &fbatch[i*WIDTH*HEIGHT*C1], WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

		cudaEventRecord(start, NULL);

				cudnnConvolutionForward(hnd_cuDNN,
					&alpha,
					desc_tsrIm, dev_fim,
					desc_wgtConv1,
					dev_fwConv1,
					desc_convolution1,
					alg_convolution1,
					dev_workspace,
					workspace_bytes,
					&beta,
					desc_tsrConv1,
					dev_fconv1);



				cudnnActivationForward(hnd_cuDNN,
					desc_relu1,
					&alpha,
					desc_tsrConv1,
					dev_fconv1,
					&beta,
					desc_tsrConv1,
					dev_fconv1);

				cudnnPoolingForward(hnd_cuDNN, desc_MaxPooling1, &alpha, desc_tsrConv1, dev_fconv1, &beta, desc_tsrMxp, dev_fmxpConv1);

				cudnnConvolutionForward(hnd_cuDNN,
					&alpha,
					desc_tsrMxp, dev_fmxpConv1,
					desc_wgtConv2,
					dev_fwConv2,
					desc_convolution2,
					alg_convolution2,
					dev_workspace2,
					workspace_bytes2,
					&beta,
					desc_tsrConv2,
					dev_fconv2);

				cudnnActivationForward(hnd_cuDNN,
					desc_relu1,
					&alpha,
					desc_tsrConv2,
					dev_fconv2,
					&beta,
					desc_tsrConv2,
					dev_fconv2);
				cudnnPoolingForward(hnd_cuDNN, desc_MaxPooling2, &alpha, desc_tsrConv2, dev_fconv2, &beta, desc_tsrMxp2, dev_fmxpConv2);

				//cublasSgemm(handle, tB, tA,
				//wB, hA, wA (hB, wA, hA )
				//&alpha,
				//dev_B, wB,
				//dev_A, wA,
				//&beta,
				//dev_C, wC);

				//Original
				//cublasSgemm(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
				//	1, 100, 24*24*32,
				//	&alpha,
				//	dev_fmxpConv2, 1,
				//	dev_fwD1, 24*24*32,
				//	&beta,
				//	dev_fd1, 1);


				//cublasSgemm(hnd_cuBLAS, CUBLAS_OP_T, CUBLAS_OP_T,
				//	100, 1,24*24*32,
				//	&alpha,
				//	
				//	dev_fwD1, 100,
				//	dev_fmxpConv2, 24*24*32,
				//	&beta,
				//	dev_fd1, 100);

				//cublasSgemm(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
				//	100, 1, 24 * 24 * 32,
				//	&alpha,
				//	dev_fwD1, 24 * 24 * 32,
				//	dev_fmxpConv2, 1,
				//	
				//	&beta,
				//	dev_fd1, 1);

				//cublasSgeam(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
				//	100, 24 * 24 * 32,
				//	&alpha,
				//	dev_fwD1, 100,
				//	&beta,
				//	dev_fwD1, 100,
				//	dev_ftwD1, 100);

				cublasSgeam(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
					24 * 24 * 32, 100,
					&alpha,
					dev_fwD1, 24 * 24 * 32,
					&beta,
					dev_fwD1, 24 * 24 * 32,
					dev_ftwD1, 24 * 24 * 32);

				cublasSgemm(hnd_cuBLAS, CUBLAS_OP_T, CUBLAS_OP_N,
					100, 1, 24 * 24 * 32,//hA, wB, hB/wA
					&alpha,
					dev_ftwD1, 24 * 24 * 32,
					dev_fmxpConv2, 24 * 24 * 32,
					&beta,
					dev_fd1, 100);



				cudnnActivationForward(hnd_cuDNN,
					desc_relu1,
					&alpha,
					desc_tsrD1,
					dev_fd1,
					&beta,
					desc_tsrD1,
					dev_fd1);

				cublasSgemm(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
					1, 100, 100,
					&alpha,
					dev_fd1, 1,
					dev_fwD2, 100,
					&beta,
					dev_fd2, 1);

				cudnnActivationForward(hnd_cuDNN,
					desc_relu1,
					&alpha,
					desc_tsrD2,
					dev_fd2,
					&beta,
					desc_tsrD2,
					dev_fd2);

				cublasSgemm(hnd_cuBLAS, CUBLAS_OP_N, CUBLAS_OP_N,
					1, 4, 100,
					&alpha,
					dev_fd2, 1,
					dev_fwD3, 100,
					&beta,
					dev_fd3, 1);

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));
		total += ms_t;

			}

		


#endif

		//printf("\nFP cuDNN + cuBLAS %f us (%1.1f ~fps)\n",  1000 * ms_t / (FRAMES*REPEAT), REPEAT*FRAMES / (ms_t / 1000.0));
		printf("\nFP cuDNN + cuBLAS (whole network) %f us (%1.1f ~fps)\n", 1000 * total / (FRAMES*REPEAT), REPEAT*FRAMES / (total / 1000.0));


	//	CUDA_ERROR(cudaMemcpy(fconv1, dev_fconv1, 32 * 96 * 96 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//	CUDA_ERROR(cudaMemcpy(fconv2, dev_fconv2, 32 * 48 * 48 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//	CUDA_ERROR(cudaMemcpy(fd3, dev_fd3, 1*4* sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//	CUDA_ERROR(cudaMemcpy(fd1, dev_fd1, 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//	CUDA_ERROR(cudaMemcpy(fmxpConv2, dev_fmxpConv2, 24*24*32 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");


		//printf("Conv1 comparison: ");
		
	//	CompareResults(gold_fconv1, fconv1, 96 * 96 * 32);
	//	printf("L1: %f\n", L1(gold_fconv1, fconv1, 96 * 96 * 32));
		
	//	printf("Conv2 comparison: ");

	//	CompareResults(gold_fconv2, fconv2, 48 * 48 * 32);
	//	printf("L1: %f\n", L1(gold_fconv2, fconv2, 48 * 48 * 32));

	//	printf("mxp2 (h) comparison: ");

	//	CompareResults(gold_fh, fmxpConv2, 24 * 24 * 32);
	//	printf("L1: %f\n", L1(gold_fh, fmxpConv2, 24 * 24 * 32));

	//	printf("cuDNN output: d1:\n");
	//	uclDisplayMatrix(fd1, 1, 100, 0);
		//printf("cuDNN output: d2:\n");
		//uclDisplayMatrix(fd3, 1, 4, 0);


	//	printf("cuDNN output: mxp2:\n");
	//	uclDisplayMatrix(fmxpConv2, 1, 33, 0);
	//	printf("cuDNN d3:\n");
	//	uclDisplayMatrix(fd3, 1, 4, 0);


//Binary net

#ifdef MEASURE_MEMCPY
		cudaDeviceSynchronize();
		cudaEventRecord(start, NULL);
		cudaDeviceSynchronize();
		//normalize image
		normalizeImage(normalizedIm, (int*)im, -127, 127);
		cudaMemcpy(dev_im, normalizedIm, WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);


		//Copy input and weights to device
		//CUDA_ERROR(cudaMemcpy(dev_im, normalizedIm, WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice), "cudaMemcpy");

		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {



				bIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_im, dev_pkIm2Col);

#if FRAMES>1
				normalizeImage(normalizedIm, (int*)&batch[i*WIDTH*HEIGHT*C1], -127, 127);
				cudaDeviceSynchronize();
				bxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_pkIm2Col, dev_conv1, dev_pkwConv1);
				cudaMemcpy(dev_im, normalizedIm, WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
#else

				bxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_pkIm2Col, dev_conv1, dev_pkwConv1);
#endif

			}

		cudaEventRecord(stop, NULL);
		(cudaEventSynchronize(stop));

		(cudaEventElapsedTime(&ms_t, start, stop));

#else

	

		//cudaMemcpy(dev_im, im, WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

		total = 0.0;

		for (int r = 0; r<REPEAT; r++)
			for (int i = 0; i < FRAMES; i++) {

				//cudaMemcpy(dev_im, im, WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_im, &batch[i*WIDTH*HEIGHT*C1], WIDTH * HEIGHT * C1 * sizeof(unsigned int), cudaMemcpyHostToDevice);
				cudaEventRecord(start, NULL);

				bIm2Col3d << < blocks_Im2Col, threads_Im2Col >> > (dev_im, dev_pkIm2Col);
				bxnorConv1 << <blocks_Conv1, threads_Conv1 >> > (dev_pkIm2Col, dev_conv1, dev_pkwConv1);

				bMaxPool1 << <blocks_MaxPool1, threads_MaxPool1 >> > (dev_conv1, dev_mxpConv1);

				bIm2Col3d_2 << < blocks_Im2Col2, threads_Im2Col2 >> > (dev_mxpConv1, dev_pkMxpConv1);
				bxnorConv2 << <blocks_Conv2, threads_Conv2 >> > (dev_pkMxpConv1, dev_pkwConv2, dev_conv2);
				bMaxPool2 << <blocks_MaxPool2, threads_MaxPool2 >> > (dev_conv2, dev_mxpConv2);

				packRowsDense1 << < blocks_pkRowsDense1, threads_pkRowsDense1 >> > (dev_mxpConv2, dev_pkH, 1);
				bxnorDense1 << <blocks_Dense1, threads_Dense1 >> > (dev_pkwD1, dev_pkH, dev_d1, 100, 24 * 24);

				packRowsDense2 << < blocks_pkRowsDense2, threads_pkRowsDense2 >> > (dev_d1, dev_pkD1, 1);
				bxnorDense2 << <blocks_Dense2, threads_Dense2 >> > (dev_pkwD2, dev_pkD1, dev_d2, 100, 4);

				packRowsDense3 << < blocks_pkRowsDense3, threads_pkRowsDense3 >> > (dev_d2, dev_pkD2, 1);
				bxnorDense3 << <blocks_Dense3, threads_Dense3 >> > (dev_pkwD3, dev_pkD2, dev_d3, 100, 4);

				cudaEventRecord(stop, NULL);
				(cudaEventSynchronize(stop));
				(cudaEventElapsedTime(&ms_t, start, stop));
				total += ms_t;
			}

		

	


#endif

		//printf("\nBinarized convolution %f us (%1.1f ~fps)\n", 1000 * ms_t / (FRAMES*REPEAT), REPEAT*FRAMES / (ms_t / 1000.0));
		printf("\nBinarized convolution (whole network) %f us (%1.1f ~fps)\n", 1000 * total / (FRAMES*REPEAT), REPEAT*FRAMES / (total / 1000.0));



		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Failed to launch 1 or more kernels: %s\n", cudaGetErrorString(cudaStatus));
		}
		else
		{
			printf("\n*All kernels launched successfully.*\n\n");
		}





#ifdef VERIFY
		//Copy results back from device

	#ifdef BINARIZED_INPUT
			CUDA_ERROR(cudaMemcpy(pkIm2Col, dev_pkIm2Col, WIDTH * HEIGHT *(3) * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");

	#else
		CUDA_ERROR(cudaMemcpy(pkIm2Col, dev_pkIm2Col, WIDTH * HEIGHT *(3) * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
			CUDA_ERROR(cudaMemcpy(Im2Col, dev_Im2Col, WIDTH * HEIGHT *(3*25) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
			CUDA_ERROR(cudaMemcpy(fIm2Col, dev_fIm2Col, WIDTH * HEIGHT *(3 * 25) * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
		//	CUDA_ERROR(cudaMemcpy(wConv1, dev_wConv1, 3*25 * 32 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	#endif

//	CUDA_ERROR(cudaMemcpy(pkH, dev_pkH, 24*24 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(conv1, dev_conv1, 32 * 96 * 96 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(fconv1, dev_fconv1, 32 * 96 * 96 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
	CUDA_ERROR(cudaMemcpy(conv1, dev_conv1, 32 * 96 * 96 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(mxpConv1, dev_mxpConv1, 32 * 48 * 48 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(conv2, dev_conv2, 32 * 48 * 48 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(mxpConv2, dev_mxpConv2, 32 * 24 * 24 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(d1, dev_d1, 100 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(pkD1, dev_pkD1, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(d2, dev_d2, 100 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(d3, dev_d3, 4 * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
	//CUDA_ERROR(cudaMemcpy(pkMxpConv1, dev_pkMxpConv1, 48 * 48 * 25 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy");
#endif
		CUDA_ERROR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	}

	else 
		fprintf(stderr, "Cannot proceed with launching kernels: %s\n", cudaGetErrorString(cudaStatus));
	



	//matrixMulCPU(gold_fconv1,fwConv1, fIm2Col,32,75,96*96);
	//printf("Conv1 comparison: ");
	//CompareResults(gold_fconv1, fconv1, 96 * 96 * 32);

		//uclDisplayMatrix(fconv1, 1, 20, 0);
		//printf("gold_fconv1:\n");
		//uclDisplayMatrix(gold_fconv1, 1, 20, 0);

		//printf("diff:\n");
		//for (int i = 0; i < 20; i++)
		//	printf("%f, ", gold_fconv1[i] - fconv1[i]);




		//printf("\n\n integer Im2Col3d comparison: ");


		//CompareResults(gold_Im2Col, Im2Col, WIDTH*HEIGHT * 3);

		//printf("integer Conv1 comparison: ");
		//CompareResults(gold_conv1, conv1, 96 * 96 * 32);

		//uclDisplayMatrixi(conv1, 1, 20, 0);
		//printf("gold_fconv1:\n");
		//uclDisplayMatrixi(gold_conv1, 1, 20, 0);

		//printf("diff:\n");
		//for (int i = 0; i < 20; i++)
		//	printf("%i, ", gold_conv1[i] - conv1[i]);
		//printf("\nim2col:");
		//uclDisplayMatrix(fIm2Col, 1, 100, 100);

		//printf("\n gold_im2col:");
		//uclDisplayMatrix(gold_fIm2Col, 1, 100, 100);



	//printf("Maxpool1 comparison: ");
	//CompareResults(gold_mxpConv1, mxpConv1, 32 * 48 * 48);

	//printf("\nIm2Col3d2 comparison: ");
	//CompareResults(gold_pkMxpConv1, pkMxpConv1, 48 * 48 * 25);

	//printf("Conv2 comparison: ");
	//CompareResults(gold_conv2, conv2, 48 * 48 * 32);

	//printf("Maxpool2 comparison: ");
	//CompareResults(gold_mxpConv2, mxpConv2, 32 * 24 * 24);

	//printf("\npkH comparison: ");
	//CompareResults(gold_pkH, pkH, 24 * 24);

	//printf("\nDense1 comparison: ");
	//CompareResults(gold_d1, d1, 100);

	//printf("Dense2 comparison: ");
	//CompareResults(gold_d2, d2, 100);

	//printf("Dense3 comparison: ");
	//CompareResults(gold_d3, d3, 4);
	//printf("Dense3 output: ");
	//uclDisplayMatrixi(d3, 1, 4, 0);
//	uclDisplayMatrixi(conv1, 1, 20, 0);
//	uclDisplayMatrixi(Im2Col, 1,75,0);


//	printf("\nCuda event timer: %f us\n", 1000 * ms_t);



	printf("\n\nFreeing host memory...\n");
	

#ifdef BINARIZED_INPUT
	CUDA_ERROR(cudaFreeHost(normalizedIm), "cudaFreeHost");
	free(im);
	free(pkIm2Col);
	free(pkwConv1);
#else
	CUDA_ERROR(cudaFreeHost(im), "cudaFreeHost");
	free(Im2Col);
	free(fwConv1);
#endif


	free(pkwConv2);
	free(pkwD1);
	free(pkwD2);
	free(pkwD3);

#ifdef VERIFY
	free(conv1);
	free(mxpConv1);
	free(pkMxpConv1);
	free(conv2);
	free(mxpConv2);
	free(pkH);
	free(d1);
	free(d2);
	free(d3);
	free(pkD1);
	free(pkD2);
#ifdef BINARIZED_INPUT
	free(gold_pkIm2Col);
#else
	free(gold_Im2Col);
#endif

	free(gold_mxpConv1);
	free(gold_pkMxpConv1);
	free(gold_mxpConv2);
	free(gold_conv1);
	free(gold_conv2);
	free(gold_pkH);
	free(gold_d1);
	free(gold_d2);
	free(gold_d3);
#endif // VERIFY

	printf("Freeing device memory...\n");
	CUDA_ERROR(cudaEventDestroy(start), "cudaEvenetDestroy");
	CUDA_ERROR(cudaEventDestroy(stop),"cudaEvenetDestroy");
	CUDA_ERROR(cudaFree(dev_im),"cudaFree");
#ifdef BINARIZED_INPUT
	CUDA_ERROR(cudaFree(dev_pkIm2Col), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkwConv1), "cudaFree");
#else

	CUDA_ERROR(cudaFree(dev_Im2Col), "cudaFree");
	CUDA_ERROR(cudaFree(dev_wConv1), "cudaFree");
#endif

	CUDA_ERROR(cudaFree(dev_conv1), "cudaFree");
	CUDA_ERROR(cudaFree(dev_mxpConv1), "cudaFree");
	CUDA_ERROR(cudaFree(dev_conv2), "cudaFree");
	CUDA_ERROR(cudaFree(dev_mxpConv2), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkMxpConv1), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkH), "cudaFree");
	CUDA_ERROR(cudaFree(dev_d1), "cudaFree");
	CUDA_ERROR(cudaFree(dev_d2), "cudaFree");
	CUDA_ERROR(cudaFree(dev_d3), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkD2), "cudaFree");

	CUDA_ERROR(cudaFree(dev_pkwConv2), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkwD1), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkwD2), "cudaFree");
	CUDA_ERROR(cudaFree(dev_pkwD3), "cudaFree");
	


	//cuDNN clean up
	printf("Clean up cuDNN...\n");
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc_tsrIm), "cudnnDestroyTensorDescriptor");
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc_tsrConv1), "cudnnDestroyTensorDescriptor");
	CUDNN_ERROR(cudnnDestroyTensorDescriptor(desc_tsrMxp), "cudnnDestroyTensorDescriptor");
	CUDNN_ERROR(cudnnDestroyFilterDescriptor(desc_wgtConv1), "cudnnDestroyFilterDescriptor");
	CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(desc_convolution1), "cudnnDestroyConvolutionDescriptor");
	CUDNN_ERROR(cudnnDestroyPoolingDescriptor(desc_MaxPooling1), "cudnnDestroyConvolutionDescriptor");
	CUDNN_ERROR(cudnnDestroyActivationDescriptor(desc_relu1), "cudnnDestroyActivationDescriptor");

	CUDNN_ERROR(cudnnDestroy(hnd_cuDNN), "cudnnDestroy");


	//cuBLAS cleanup
	printf("Clean up cuBLAS...\n");
	cublasDestroy(hnd_cuBLAS);


	
	printf("Reset device...\n");
	CUDA_ERROR(cudaDeviceReset(), "cudaDeviceReset");

	pause();

	return 0;
}



