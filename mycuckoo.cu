// THIS CODE IMPLEMENTS CUCKOO'S HASHING USING CUDA
#include<iostream>
#include<fstream>
#include<string>
using namespace std;

struct Entry{
	int key;
	int value;
	};


// Prefix Scan using serial Algorithm
void serialScan(unsigned *start,unsigned *count,int n){
	start[0] = 0;
	for (int i = 1;i<n;i++)
			start[i] = start[i-1] + count[i-1];	
	}
		
void giveout(string str,unsigned *var,int size){
ofstream fo;
fo.open(str.c_str());
for (int i = 0;i<size;i++)
		fo<<var[i]<<endl;
fo.close();
}
			
// Prefix Scan in Cuda (a work efficient algorithm by Mark Harris NVIDIA Corp.)
__global__
void prescan(unsigned *g_odata,unsigned *g_idata, int n){
	extern __shared__ unsigned temp[];
// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2*thid] = g_idata[2*thid];
// load input into shared memory
	temp[2*thid+1] = g_idata[2*thid+1];
	for( int d = n>>1; d > 0; d >>= 1){
	// build sum in place up the tree
		__syncthreads();
	if(thid < d){
		int ai = offset*(2*thid+1)-1;
		int bi = offset*(2*thid+2)-1;
		temp[bi] += temp[ai];
		}
	offset *= 2;
	}
	if(thid == 0) { temp[n - 1] = 0; }
// clear the last element
	for(int d = 1; d < n; d *= 2){
// traverse down tree & build scan
		offset >>= 1;
		__syncthreads();
		if(thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			unsigned t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
			}
	}
	__syncthreads();
	g_odata[2*thid] = temp[2*thid];
// write results to device memory
	g_odata[2*thid+1] = temp[2*thid+1];
} 


// Funtion that hashes key-value pairs in the bucket
__global__ void
phase1(unsigned* count, Entry* entry,unsigned *offset, int n,int nentry,int a, int b, int* global_flag,int max_per_bucket ){
// for all k belong to  keys, in parallel do 
	int i = blockDim.x * blockIdx.x +threadIdx.x;
	unsigned kPrime = 1900813;
// 	 compute g(k) to determine bucket b_k containing k
	int indexd = ((a * entry[i].key + b) % kPrime) % n;
//	 atomically increment count[b_k], learning internal offset[k]
	if (i<nentry){
		offset[i] = atomicAdd(&count[indexd],1) -1;
		__syncthreads();
		if(count[indexd]>max_per_bucket)
			global_flag[indexd] = -1;
// end for
	}
}


// Funtion used to store in phase1
__global__ void
store(Entry* shuffled, Entry* entry, unsigned* start, unsigned* offset, int a, int b, int n,int* test){
// for all key value pairs, do in parallel
// store (k,v) in shuffled[] at index start[b_k]+offset[k]
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned kPrime = 1900813;

	int index = ((a * entry[i].key + b) % kPrime) % n;
	shuffled[offset[i] + start[index]] = entry[i];
}
	
	
__global__
void phase2(unsigned *random_numbers,Entry *shuffled, Entry *cuckoo, unsigned *start,unsigned *seed, int subtable_size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;			// Thread
	// Define constants for 3 subtables
	__shared__ int j ;
	__shared__ unsigned max_seeds;
	__shared__ int *sh_subtable_1, *sh_subtable_2, *sh_subtable_3;	
	__shared__ unsigned *sh_hash_built;
		
	if (threadIdx.x == 0){
			j = blockDim.x * blockIdx.x;				// Block
			max_seeds = 20;
			}
	__syncthreads();
	
	int has_pair;
	unsigned seed_index = 0;
		
	if (i < start[j+1])
			has_pair = 1;
	else
			has_pair = 0;
	
	int in_subtable = -1;
	unsigned index_x, index_y, index_z;
	do{
// 		generate the h_i(k) using one seed
		short random_number_seed = random_numbers[seed_index++];

		unsigned constants_0 = random_number_seed ^ 0xffff;
		unsigned constants_1 = random_number_seed ^ 0xcba9;
		unsigned constants_2 = random_number_seed ^ 0x7531;
		unsigned constants_3 = random_number_seed ^ 0xbeef;
		unsigned constants_4 = random_number_seed ^ 0xd9f1;
		unsigned constants_5 = random_number_seed ^ 0x337a;
		
		unsigned kPrime = 1900813;

		unsigned key = shuffled[i].key;
//		pre-compute h_i(k) for all keys k	
		index_x = ((constants_0 * key + constants_1)%kPrime) % subtable_size;
		index_y = ((constants_2 * key + constants_3)%kPrime) % subtable_size;
		index_z = ((constants_4 * key + constants_5)%kPrime) % subtable_size;
	
		int max_iterations = 30;	
		for(int iteration = 1; iteration <= max_iterations; ++iteration){
			if (threadIdx.x == 0)
				*sh_hash_built = 0;
//		while any k is uninserted and failure is unlikely do 	
//			write all uninserted k into slot g_1(k)
			if (has_pair && in_subtable == -1){
				sh_subtable_1[index_x] = key;
				in_subtable = 1;
				}
//			synchronize threads
			__syncthreads();

//			check subtable T_1  and write all uninserted key into slot g_2(k)
			if ( has_pair && in_subtable == 1 && sh_subtable_1[index_x] != key) {
				sh_subtable_2[index_y] = key;
				in_subtable = 2;
				}
//			synchronize threads			
			__syncthreads();

//			check subtable T_2  and write all uninserted key into slot g_3(k)
			if ( has_pair && in_subtable == 2 && sh_subtable_2[index_y] != key) {
				sh_subtable_3[index_z] = key;
				in_subtable = 3;
				}
//			synchronize threads			
			__syncthreads();	

//			check subtable T_3
			if ( has_pair && in_subtable == 3 && sh_subtable_3[index_z] != key) {
				*sh_hash_built = 1;
				in_subtable = -1;
				}
//			synchronize threads
			__syncthreads();
			
//			evaluate if hash_built
			if (*sh_hash_built == 0) 
					break;
			
			__syncthreads();
		}
	}while(*sh_hash_built && seed_index < max_seeds);

//	write subtables T_1, T_2 and T_3 into cuckoo[] and 
	if (in_subtable == 1 && has_pair){
			int position = 3 * j  * subtable_size + index_x;
			cuckoo[position].key 	= shuffled[i].key;
			cuckoo[position].value	= shuffled[i].value;
			}
	else if (in_subtable == 2 && has_pair){
			int position = (3 * j +1) * subtable_size + index_y;
			cuckoo[position].key 	= shuffled[i].key;
			cuckoo[position].value	= shuffled[i].value;
			}
	else if (in_subtable == 3 && has_pair){
			int position = (3 * j  +2) * subtable_size + index_z;
			cuckoo[position].key 	= shuffled[i].key;
			cuckoo[position].value	= shuffled[i].value;
			}
//  Write the final seed used into seed[b]: only one thread from each block
	if (threadIdx.x == blockDim.x * blockIdx.x + 0)
			seed[j] = random_numbers[seed_index-1];
}

int ceiling(float a){
	int b = (int)a + 1 ;
	return b;
	}
	
/**********************MAIN STARTS HERE******************************************************************/
int main(){

cudaError_t err = cudaSuccess;
// Generate key value pairs

cout<<"\nGenerating key-value pairs..\n";
int N = 10000;
Entry *entry;
entry = new Entry[N];
for (int i = 0;i < N; i++){
		entry[i].key = i+1;
		entry[i].value = rand()%100;
		}

cout<<"Initializing Phase I of hashing...\n";
/********************** Phase I *******************************/
// INPUT: key value pairs and size of the array
// OUTPUT: shuffled[] and start[]

// Estimate the number of buckets
float occupancy =		0.8;
int size_table =		N;
int max_per_bucket =	512;
int number_of_buckets = size_table /(occupancy*max_per_bucket)+1;
bool bucket_overflow = 0;
cout<<number_of_buckets;

// Allocate output arrays and scratch space
Entry *shuffled;
shuffled = new Entry[size_table];

unsigned *start;
start = new unsigned[number_of_buckets];
unsigned *count;
count = new unsigned[number_of_buckets];
unsigned *offset;
offset = new unsigned[number_of_buckets];

int threadsPerBlock = 512;
int num_blocks = number_of_buckets+1;
int a,b;
int ctr = 0;

cout<<"Hashing on "<<num_blocks<<" blocks with "<<threadsPerBlock<<" threads per block...\n";
// Repeat
do{
// Set all bucket size counters count[i] = 0;
	cout<<"Setting bucket size counters to zero\n";
	for (int i = 0;i < number_of_buckets;i++)
			count[i] = 0;	

// Constants defining hash functions
	cout<<"Defining constants for hash function\n";
	a = rand()%10;
	b = rand()%10;
	cout<<"a,b = "<<a<<"\t"<<b<<endl;
	
// Copy variables to device
	cout<<"Allocating memory for count[], entry[] and offset[]\n";
	unsigned *d_count = NULL;
	size_t size_count = number_of_buckets*sizeof(unsigned);
	
	err = cudaMalloc((void**)&d_count,size_count);
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to d_count[]\n";
		exit(0);
		}
		
	Entry *d_entry = NULL;
	size_t size_entry = N*sizeof(Entry);
	
	err = cudaMalloc((void**)&d_entry,size_entry);
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to d_entry[]\n";
		exit(0);
		}
		
	unsigned *d_offset = NULL;
	
	err = cudaMalloc((void**)&d_offset,N*sizeof(unsigned));	
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to d_offset[]\n";
		exit(0);
		}

	cout<<"\nCopying count[] and entry[] for sorting into buckets, from host to CUDA device...\n";
	
	err = cudaMemcpy(d_entry,entry,size_entry,cudaMemcpyHostToDevice);
	if(err!= cudaSuccess){
		cout<<"Could not copy entry[] to device!!!\n";
		exit(0);
		}
	
	err = cudaMemcpy(d_count,count,size_count,cudaMemcpyHostToDevice);
	if(err!= cudaSuccess){
		cout<<"Could not copy count[] to device!!!\n";
		exit(0);
		}
		
	cout<<"Defining global_flag[]\n";
	int *global_flag = new int[N];
	int *d_global_flag;
	
	err = cudaMalloc((void**)&d_global_flag,N*sizeof(int));
	if (err != cudaSuccess){
		cout<<"Could not allocate memory for global_flag\n";
		exit(0);
		}
		
	cout<<"Calling phase1()\n";
// Calling phase1() 
	phase1<<<num_blocks,threadsPerBlock>>>(d_count,d_entry,d_offset,number_of_buckets,N,a,b,d_global_flag,max_per_bucket);

// Copy back from device
	cout<<"Copying back global_flag[], count[] and offset[] from device\n";
	err = cudaMemcpy(global_flag,d_global_flag,N*sizeof(int),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Could not copy global_flag back to host\n";cout<<cudaGetErrorString(err)<<endl;
		exit(0);
		}

	err = cudaMemcpy(count,d_count,size_count,cudaMemcpyDeviceToHost);
	if (err !=  cudaSuccess){
		cout<<"Could not copy count[] back to host!!!\n";
		exit(0);
		}
	
	err = cudaMemcpy(offset,d_offset,N*sizeof(unsigned),cudaMemcpyDeviceToHost);
	if (err !=  cudaSuccess){
		cout<<"Could not copy offset[] back to host!!!\n";
		exit(0);
		}	
// Checking if the bucket has overflown

	cout<<"Checking for overflow of bucket\n";
	bucket_overflow = 0;
	for (int i = 0;i < N; i++){
			if (global_flag[i] == -1){
				cout<<"Bucket overfill... Rehashing using new hash function\n";
				bucket_overflow = 1;
				ctr++;
				break;
				}
			}
 }while(bucket_overflow == 1 && ctr<10);	
 
 if (ctr >10) {
 		cout<<"Phase I hashing failed. Exiting...\n";
 		exit(0);
 		}

/* cout<<"\nCOUNT = ";
for (int k =0;k<number_of_buckets;k++)
		cout<<count[k]<<"\t";*/
// Perform prefix sum on count[] to determine start[] in SERIAL 

cout<<"Calling serialScan\n";
	serialScan(start,count,number_of_buckets);

/*
// Perform prefix sum on count[] to determines start[] using CUDA
	cout<<"Allocating memory for count[] and start[] on device\n";
	unsigned *g_count = NULL;
	unsigned *g_start = NULL;
	err = cudaMalloc((void**)&g_count,number_of_buckets*sizeof(unsigned));
	if (err!= cudaSuccess){
		cout<<"Could not allocate memory for count[] for prefix scan on CUDA device\n";
		exit(0);
		}
	err = cudaMalloc((void**)&g_start,number_of_buckets*sizeof(unsigned));
	if (err!= cudaSuccess){
		cout<<"Could not allocate memory for start[] for prefix scan on CUDA device\n";
		exit(0);
		}
	
	cout<<"Copying count[] to the device for prefix scan\n";
	err = cudaMemcpy(g_count,count,number_of_buckets*sizeof(unsigned),cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Could not count copy from host to device	for prefix scan\n";
		exit(0);
		}
		
	// Calling prescan()
	cout<<"Calling prescan()\n";
	prescan<<<num_blocks,threadsPerBlock>>>(g_start,g_count,number_of_buckets);
	
	cout<<"Copying back start[] to host\n";
	err = cudaMemcpy(start,g_start,number_of_buckets*sizeof(unsigned),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Could not copy start[] from CUDA device\n"<<cudaGetErrorString(err)<<endl;
		exit(0);
		}
	cout<<"Freeing memory occupied by count[] on device\n";	
	cudaFree(g_count);
	*/
	
// for all key-value pairs (k,v) in parallel do 
// store (k,c) in shuffled[] at index start[b_k] + offset[k]
// end for

	string s ="offset.txt";
	giveout(s,offset,N);
	s = "start.txt";
	giveout(s,start,number_of_buckets);
	cout<<"Allocating memory for start[], shuffled[] and entry[]\n";
	Entry* d_shuffled;
	err = cudaMalloc((void**)&d_shuffled,N*sizeof(Entry));
	if (err!= cudaSuccess){
		cout<<"Could not allocate memory to shuffled[] on CUDA device\n";
		exit(0);
		}
	
	Entry *g_entry;
	err = cudaMalloc((void**)&g_entry,N*sizeof(Entry));
	if (err!= cudaSuccess){
		cout<<"Could not allocate memory to g_entry[] on CUDA device\n";
		exit(0);
		}
	size_t size_count = number_of_buckets*sizeof(unsigned);
	unsigned *g_start = NULL;
	err = cudaMalloc((void**)&g_start,size_count);
	if (err!=cudaSuccess){
		cout<<"\nCould not allocate memory to g_start[] on CUDA device\n";
		exit(0);
		}
	
	cout<<"Initializing 	
	cout<<"Copying entry[], offset[] and start[] onto cuda device\n";
	err = cudaMemcpy(g_entry, entry, N*sizeof(Entry), cudaMemcpyHostToDevice);
	if (err!= cudaSuccess){
		cout<<"Could not copy g_entry[] to CUDA device\n";
		exit(0);
		}
	
	err = cudaMemcpy(g_start,start, size_count,cudaMemcpyHostToDevice);
	if (err!=cudaSuccess){
		cout<<"\nCould not copy start[] to CUDA device\n";
		exit(0);
		}

	cout<<"Allocating memory to offset[]\n";
	unsigned *d_offset = NULL;
	err = cudaMalloc((void**)&d_offset,N*sizeof(unsigned));	
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to d_offset[]\n";
		exit(0);
		}
	
	cout<<"Copying offset to cuda device\n";	
	err = cudaMemcpy(d_offset,offset,N*sizeof(unsigned),cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Could not copy offset to cuda device\n";
		exit(0);
		}
	
	cout<<"Initializing shuffle and copying to CUDA device\n";
	for(int i = 0;i<N;i++)
		shuffled[i] = 0xffffff;	
	
	err = cudaMemcpy(d_shuffled,shuffled,N*sizeof(Entry),cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Could not copy shuffle to CUDA device\n";
		exit(0);
		}
		
	// Calling store()
	cout<<"Calling store()\n";
	store<<<num_blocks,threadsPerBlock>>>(d_shuffled,g_entry, g_start, d_offset,a,b,number_of_buckets);
	
	cout<<"Copying shuffled[] back to host\n";
	err = cudaMemcpy(shuffled,d_shuffled,N*sizeof(Entry),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Could not copy shuffled[] from CUDA device\n"<<cudaGetErrorString(err);
		exit(0);
		}

// We now have start[] and shuffled[]

cout<<"\nBeginning Phase II\n";
/***********************************Phase II********************/
// INPUT: shuffled[] and start[]
// OUPUT: seed[] and cuckoo[]

// initialize cuckoo[] array
	cout<<"Create and initialize cuckoo[]\n";
	Entry *cuckoo;
	int cuckoo_size = ceiling(3*(float)(N/4 + N/8));
	cuckoo = new Entry[cuckoo_size];
	for (int i = 0; i < cuckoo_size; i++)
		cuckoo[i].key = 0xffffffff;
	
	unsigned *seed;
	seed = new unsigned [number_of_buckets];
	
	unsigned max_seeds = 10;	
	unsigned *random_numbers;
	random_numbers = new unsigned [max_seeds];
	for (int i = 0;i<max_seeds;i++)
		random_numbers[i] = rand()%100 +1;
	
	// Copy random_numbers
	cout<<"Create, allocate memory and copy random_numbers[] to device\n";
	unsigned *d_random_numbers;
	err = cudaMalloc((void**)&d_random_numbers,max_seeds*sizeof(unsigned));
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to random_numbers[] on device\n";
		exit(0);
		}
	err = cudaMemcpy(d_random_numbers, random_numbers, max_seeds*sizeof(unsigned), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		cout<<"Could not copy random_numbers[] to CUDA device\n";
		exit(0);
		}
	
	cout<<"Allocate memory to cuckoo[] and seed[] on device\n";	
	Entry *d_cuckoo;
	err = cudaMalloc((void**)&d_cuckoo, cuckoo_size*sizeof(Entry));
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to cuckoo[] on device\n";
		exit(0);
		}
	
	unsigned *d_seed;
	err = cudaMalloc((void**)&d_seed, number_of_buckets*sizeof(unsigned));
	if (err != cudaSuccess){
		cout<<"Could not allocate memory to seed[] on device\n";
		exit(0);
		}
	
	// Evaluating subtable size

	int subtable_size = ceiling((float)cuckoo_size/(number_of_buckets*3));
	cout<<"Using subtable size as "<<subtable_size<<endl;
		
	// Evaluating phase 2
	cout<<"Calling phase2()\n";	
	phase2<<<num_blocks,threadsPerBlock>>>(d_random_numbers, d_shuffled, d_cuckoo, start, d_seed, subtable_size);
	
	cout<<"Copying back cuckoo[] and seed[]\n";
	err = cudaMemcpy(cuckoo,d_cuckoo, max_seeds*sizeof(unsigned), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Could not copy random_numbers[] back to host\n";
		exit(0);
		}
		
	err = cudaMemcpy(seed,d_seed, number_of_buckets*sizeof(unsigned),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		cout<<"Could not copy seed[] back to host\n";
		exit(0);
		}
	cout<<"\n Hash table successfully created!!!\nFreeing CUDA memory\n";	
	cudaFree(d_seed);
	cudaFree(d_cuckoo);
	cudaFree(d_random_numbers);
	cudaFree(d_shuffled);
	
}

/******************************************** MAIN ENDS HERE******************************************/
