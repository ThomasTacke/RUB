/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "sha256gpu.h"
#include "cuda.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include "sha256cpu.h"

using namespace std;
using std::fixed;

#define maxLength 8
#define minLength 1

void clearScreen() {
#ifdef _WIN32
	std::system ( "CLS" );
#else
	// Assume POSIX
	std::system("clear");
#endif
}

__constant__ int K[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
		0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
		0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138,
		0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624,
		0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f,
		0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
__constant__ uint32_t h_values[] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372,
		0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

__constant__ char usedAlphabet[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
		'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
		'w', 'x', 'y', 'z' };
__constant__ uint16_t usedAlphabetSize = sizeof(usedAlphabet);

__global__ void kernel(uint32_t *target, char *dev_result,
		char *devPositionInUsedAlphabet, uint16_t iterations_per_thread,
		char *dev_match_found, uint8_t stringLength, uint8_t usedAlphabetSize) {
	uint8_t stringLengthTemp = stringLength;
	uint32_t S0;
	uint32_t S1;
	unsigned char block[64];
	uint32_t W[64];
	bool match = false;
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;

	/*copy starting positionInUsedAlphabet*/
	char attackedString[maxLength + 1];
	char positionInUsedAlphabet[maxLength];
	for (int i = 0; i < stringLength + 1; i++) {
		attackedString[i] = 0;
	}
	for (int i = 0; i < stringLength; i++) {
		positionInUsedAlphabet[i] = devPositionInUsedAlphabet[i];
	}

	/*set the position for each thread individually*/
	//working wieder EINBAUEN falls neuer Ansatz nicht geht
	/*for(int j=stringLength-1;j>=0;j--)
	 {
	 uint64_t temp=1;
	 //manual pow for pow(usedAlphabetSize, stringLength-j)
	 for(int i=1;i<stringLength-j;i++)
	 {
	 temp=temp*usedAlphabetSize;
	 }
	 positionInUsedAlphabet[j]=positionInUsedAlphabet[j]+((uint32_t)(iterations_per_thread*thread_number/ temp) %usedAlphabetSize);
	 //positionInUsedAlphabet[j]=positionInUsedAlphabet[j]+((uint32_t)(iterations_per_thread*thread_number/ pow( (double) usedAlphabetSize, (double) (stringLength-j) )) %usedAlphabetSize);
	 }

	 /*set the position for each thread individually*/

	float temp = iterations_per_thread * thread_number;
	for (int j = stringLength - 1; j >= 0; j--) {
		positionInUsedAlphabet[j] = positionInUsedAlphabet[j]
				+ (uint32_t(temp) % usedAlphabetSize);
		temp = temp / usedAlphabetSize;
		if (temp < 1)
			break;
	}
	positionInUsedAlphabet[0] = positionInUsedAlphabet[0]
			+ (uint32_t(temp) % usedAlphabetSize);
	if (temp > usedAlphabetSize) {
		temp = iterations_per_thread * thread_number;
		return;
	}
	for (int j = stringLength - 1; j > 0; j--) {
		while (positionInUsedAlphabet[j] >= usedAlphabetSize) {
			positionInUsedAlphabet[j] -= usedAlphabetSize;
			positionInUsedAlphabet[j - 1]++;
		}
	}

	//temporary variables, values irrelevant*/
	uint32_t hashes[8];

	for (int j = 0; j < iterations_per_thread; j++) {

		for (int i = stringLength - 1; i > 0; i--) {
			if (positionInUsedAlphabet[i] >= usedAlphabetSize) {
				positionInUsedAlphabet[i] = 0;
				positionInUsedAlphabet[i - 1]++;
			}
		}
		if (positionInUsedAlphabet[0] >= usedAlphabetSize)
			break; //if(positionInUsedAlphabet[maxLength-stringLength]>usedAlphabetSize-1)break;
		for (int i = 0; i < stringLength; i++) {
			attackedString[i] = usedAlphabet[positionInUsedAlphabet[i]];
		}

		//Message Copy
		for (int i = 0; i < stringLength; i++) {
			block[i] = attackedString[i];
		}

		//SHA2-begin
		//Padding
		block[stringLength] = 0x80;	//append a 1 (1000.0000)

		for (int i = stringLength + 1; i < 56; i++)	//fill with 0 except the last 8-byte
				{
			block[i] = 0x0;
		}

		stringLength = stringLength * 8;
		//append the length of the attackedString in big endianes
		for (int i = 0; i < 8; i++) {
			block[63 - i] = stringLength;
			stringLength = stringLength >> 8;
		}

		uint32_t a = h_values[0];
		uint32_t b = h_values[1];
		uint32_t c = h_values[2];
		uint32_t d = h_values[3];
		uint32_t e = h_values[4];
		uint32_t f = h_values[5];
		uint32_t g = h_values[6];
		uint32_t h = h_values[7];
		uint32_t T1;
		uint32_t T2;

		//compute W
		for (int i = 0; i < 16; i++) {
			W[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16)
					| (block[i * 4 + 2] << 8) | (block[i * 4 + 3]);
		}
		for (int i = 16; i < 64; i++) {
			S0 =
					(((W[i - 15] >> 7) | (W[i - 15] << 25))
							^ ((W[i - 15] >> 18) | (W[i - 15] << 14))
							^ (W[i - 15] >> 3));
			S1 = (((W[i - 2] >> 17) | (W[i - 2] << 15))
					^ ((W[i - 2] >> 19) | (W[i - 2] << 13)) ^ (W[i - 2] >> 10));
			W[i] = (S0 + S1 + W[i - 7] + W[i - 16]) & 0xFFFFFFFF;
		}

		//Hash
		for (int i = 0; i < 64; i++) {
			T1 = (h + ((e & f) ^ ((~e) & g) /*Ch(e,f,g)*/)
					+ (((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21))
							^ ((e >> 25) | (e << 7)) /*Sigma1(e)*/) + K[i]
					+ W[i]) & 0xFFFFFFFF;
			T2 = ((((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19))
					^ ((a >> 22) | (a << 10)) /*Sigma0(a)*/)
					+ ((a & b) ^ (a & c) ^ (b & c) /*Maj(a,b,c)*/))
					& 0xFFFFFFFF;
			h = g;
			g = f;
			f = e;
			e = (d + T1) & 0xFFFFFFFF;
			d = c;
			c = b;
			b = a;
			a = (T1 + T2) & 0xFFFFFFFF;
		}

		hashes[0] = (a + h_values[0]) & 0xFFFFFFFF;
		hashes[1] = (b + h_values[1]) & 0xFFFFFFFF;
		hashes[2] = (c + h_values[2]) & 0xFFFFFFFF;
		hashes[3] = (d + h_values[3]) & 0xFFFFFFFF;
		hashes[4] = (e + h_values[4]) & 0xFFFFFFFF;
		hashes[5] = (f + h_values[5]) & 0xFFFFFFFF;
		hashes[6] = (g + h_values[6]) & 0xFFFFFFFF;
		hashes[7] = (h + h_values[7]) & 0xFFFFFFFF;

		//SHA2-end

		match = true;
		for (int i = 0; i < 8; i++) {
			if (hashes[i] != target[i]) {
				match = false;
			}
		}
		if (match) {
			for (int i = 0; i < sizeof(dev_result); i++) {
				dev_result[i] = 0;
			}
			for (int i = 0; i < stringLengthTemp; i++) {
				dev_result[i] = attackedString[i];
			}
			dev_match_found[0] = 1;
			return;
		}
		stringLength = stringLengthTemp;
		positionInUsedAlphabet[stringLength - 1]++;
	}

}

/*
 Start the kernel with 100 iterations and measure the time. Then adjust the iterations_per_thread in a way that the kernel will take 2 seconds
 */
int get_iterations_per_thread(uint16_t number_of_blocks,
		uint16_t threads_per_block) {
	uint32_t impossible_target[] = { 0x183bddb1, 0xf21ab681, 0x3bd9b6b7,
			0x907bfe76, 0x49036ad7, 0xbc75cd39, 0x6df352fe, 0xd7dcb135 };
	uint32_t *dev_impossible_target;
	char *dev_result;
	char *dev_match_found;
	char *devPositionInUsedAlphabet;
	char match_found[1];
	float elapsedTime_kernel;

	char usedAlphabet[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
			'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
			'x', 'y', 'z' };
	uint8_t usedAlphabetSize = sizeof(usedAlphabet);
	char *dev_usedAlphabet;

	cudaEvent_t start_kernel, stop_kernel;
	HANDLE_ERROR(cudaEventCreate(&start_kernel));
	HANDLE_ERROR(cudaEventCreate(&stop_kernel));

	HANDLE_ERROR(cudaMalloc((void**)&dev_result,maxLength));
	HANDLE_ERROR(cudaMalloc((void** )&dev_match_found, 1));
	HANDLE_ERROR(
			cudaMemcpy(dev_match_found, match_found, 1,
					cudaMemcpyHostToDevice));

	HANDLE_ERROR(
			cudaMalloc((void** )&dev_impossible_target, sizeof(uint32_t) * 8));
	HANDLE_ERROR(
			cudaMemcpy(dev_impossible_target, impossible_target,
					sizeof(uint32_t) * 8, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void** )&dev_usedAlphabet, usedAlphabetSize));
	HANDLE_ERROR(
			cudaMemcpy(dev_usedAlphabet, usedAlphabet, usedAlphabetSize,
					cudaMemcpyHostToDevice));

	char * startString = (char*) malloc(sizeof(char) * (11));

	HANDLE_ERROR(cudaEventRecord(start_kernel, 0));

	kernel<<<number_of_blocks, threads_per_block>>>(dev_impossible_target,
			dev_result, devPositionInUsedAlphabet, 2000, dev_match_found, 10,
			usedAlphabetSize);

	HANDLE_ERROR(cudaEventRecord(stop_kernel, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop_kernel));
	HANDLE_ERROR(
			cudaEventElapsedTime(&elapsedTime_kernel, start_kernel,
					stop_kernel));

	cout << endl << elapsedTime_kernel << "=> "
			<< 110 * 1000 * 2 / elapsedTime_kernel / 4;	//factor 4 is because of 4 streams  90 klappt f�r release, f�r profiling wird auf 70 reduziert werden

	/*string schwts = "";
	 cin>>schwts;*/

	cudaFree(dev_usedAlphabet);
	cudaFree(dev_result);
	cudaFree(dev_match_found);
	cudaFree(dev_impossible_target);

	return (int) (110 * 1000 * 2 / elapsedTime_kernel / 4);	//factor 4 is because of 4 streams  90 klappt f�r release, f�r profiling wird auf 70 reduziert werden
}

void sha256bruteforce(uint32_t *target) {
	//Get cuda device prop
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	uint32_t max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
	uint32_t max_blocks_per_SM = 16;
	if (prop.major < 3)
		max_blocks_per_SM = 8;
	uint32_t threads_per_block = max_threads_per_SM / max_blocks_per_SM;
	uint32_t number_of_blocks = max_blocks_per_SM * prop.multiProcessorCount;
	uint32_t *dev_target;
	char result[maxLength + 1];
	char *dev_result;
	char *dev_match_found;
	char match_found[1];

	char usedAlphabet[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
			'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
			'x', 'y', 'z' };
	uint8_t usedAlphabetSize = sizeof(usedAlphabet);
	char *dev_usedAlphabet;
	char *devPositionInUsedAlphabet;
	match_found[0] = 0;
	cudaStream_t stream0, stream1, stream2, stream3;

	// capture the start time
	cudaEvent_t start_kernel, start_overall, stop_kernel, stop_overall;
	HANDLE_ERROR(cudaEventCreate(&start_kernel));
	HANDLE_ERROR(cudaEventCreate(&stop_kernel));
	HANDLE_ERROR(cudaEventCreate(&start_overall));
	HANDLE_ERROR(cudaEventCreate(&stop_overall));

	HANDLE_ERROR(cudaMalloc((void** )&dev_result, 16));
	HANDLE_ERROR(cudaMalloc((void** )&dev_match_found, 1));
	HANDLE_ERROR(
			cudaMemcpy(dev_match_found, match_found, 1,
					cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void** )&dev_target, sizeof(uint32_t) * 8));
	HANDLE_ERROR(
			cudaMemcpy(dev_target, target, sizeof(uint32_t) * 8,
					cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void** )&dev_usedAlphabet, usedAlphabetSize));
	HANDLE_ERROR(
			cudaMemcpy(dev_usedAlphabet, usedAlphabet, usedAlphabetSize,
					cudaMemcpyHostToDevice));

	uint32_t iterations_per_thread = 9000;
	//uint32_t iterations_per_thread=get_iterations_per_thread(number_of_blocks,threads_per_block);	

	float elapsedTime_overall;

	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	HANDLE_ERROR(cudaStreamCreate(&stream2));
	HANDLE_ERROR(cudaStreamCreate(&stream3));
	HANDLE_ERROR(cudaEventRecord(start_overall, 0));

	double temp = number_of_blocks * threads_per_block * iterations_per_thread;
	cout << "Iterations per thread: " << iterations_per_thread << endl
			<< "threads per block: " << threads_per_block << endl
			<< "number of blocks: " << number_of_blocks << endl
			<< "=>steps per kernel: "
			<< number_of_blocks * threads_per_block * iterations_per_thread
			<< endl;
	for (int i = minLength; i <= maxLength; i++)//i= length of the attacked String
			{
		cout << endl << endl << "Length " << i << endl;
		char * startString = (char*) malloc(sizeof(char) * (i + 1));//i = length of the string
		uint8_t * positionInUsedAlphabet = (uint8_t*) malloc(sizeof(char) * i);
		//fill the startString with the first character in the used alphabet
		for (int j = 0; j < i; j++) {
			positionInUsedAlphabet[j] = 0;
			startString[j] = usedAlphabet[0];
		}
		startString[i] = 0;	//terminate with 0

		HANDLE_ERROR(
				cudaMalloc((void** )&devPositionInUsedAlphabet,
						sizeof(uint8_t) * i));
		//HANDLE_ERROR( cudaEventRecord( start_kernel, 0 ) );
		do {

			HANDLE_ERROR(
					cudaMemcpy(devPositionInUsedAlphabet,
							positionInUsedAlphabet, sizeof(uint8_t) * i,
							cudaMemcpyHostToDevice));

			cout << "attacked String: " << startString << endl;
			//HANDLE_ERROR( cudaEventRecord( start_kernel, 0 ) );
			if (match_found[0] == 1) {
				break;
			}
			kernel<<<number_of_blocks, threads_per_block, 0, stream0>>>(
					dev_target, dev_result, devPositionInUsedAlphabet,
					iterations_per_thread, dev_match_found, i,
					usedAlphabetSize);
			/*HANDLE_ERROR( cudaEventRecord( stop_kernel, 0 ) );
			 HANDLE_ERROR( cudaEventSynchronize( stop_kernel ) );
			 HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime_kernel,start_kernel, stop_kernel ) );
			 cout<<elapsedTime_kernel<<endl;*/
			//change the attacked string for the second kernel
			for (int j = i - 1; j > 0; j--) {
				positionInUsedAlphabet[j] = positionInUsedAlphabet[j]
						+ (uint32_t(temp) % usedAlphabetSize);
				temp = temp / usedAlphabetSize;
				if (temp < 1)
					break;
			}
			positionInUsedAlphabet[0] = positionInUsedAlphabet[0]
					+ (uint32_t(temp) % usedAlphabetSize);
			if (temp > usedAlphabetSize) {
				temp = number_of_blocks * threads_per_block
						* iterations_per_thread;
				break;
			}
			for (int j = i - 1; j > 0; j--) {
				while (positionInUsedAlphabet[j] >= usedAlphabetSize) {
					positionInUsedAlphabet[j] -= usedAlphabetSize;
					positionInUsedAlphabet[j - 1]++;
				}
			}
			temp = number_of_blocks * threads_per_block * iterations_per_thread;
			for (int j = 0; j < i; j++) {
				startString[j] = usedAlphabet[positionInUsedAlphabet[j]];
			}
			HANDLE_ERROR(
					cudaMemcpy(devPositionInUsedAlphabet,
							positionInUsedAlphabet, sizeof(uint8_t) * i,
							cudaMemcpyHostToDevice));
			cout << "attacked String: " << startString << endl;
			if (positionInUsedAlphabet[0] >= usedAlphabetSize)
				break;
			match_found[0] = 0;
			HANDLE_ERROR(
					cudaMemcpy(match_found, dev_match_found, 1,
							cudaMemcpyDeviceToHost));
			if (match_found[0] == 1) {
				break;
			}
			kernel<<<number_of_blocks, threads_per_block, 0, stream1>>>(
					dev_target, dev_result, devPositionInUsedAlphabet,
					iterations_per_thread, dev_match_found, i,
					usedAlphabetSize);
			//change the attacked string for the third kernel
			for (int j = i - 1; j > 0; j--) {
				positionInUsedAlphabet[j] = positionInUsedAlphabet[j]
						+ (uint32_t(temp) % usedAlphabetSize);
				temp = temp / usedAlphabetSize;
				if (temp < 1)
					break;
			}
			positionInUsedAlphabet[0] = positionInUsedAlphabet[0]
					+ (uint32_t(temp) % usedAlphabetSize);
			if (temp > usedAlphabetSize) {
				temp = number_of_blocks * threads_per_block
						* iterations_per_thread;
				break;
			}
			for (int j = i - 1; j > 0; j--) {
				while (positionInUsedAlphabet[j] >= usedAlphabetSize) {
					positionInUsedAlphabet[j] -= usedAlphabetSize;
					positionInUsedAlphabet[j - 1]++;
				}
			}
			temp = number_of_blocks * threads_per_block * iterations_per_thread;
			for (int j = 0; j < i; j++) {
				startString[j] = usedAlphabet[positionInUsedAlphabet[j]];
			}
			HANDLE_ERROR(
					cudaMemcpy(devPositionInUsedAlphabet,
							positionInUsedAlphabet, sizeof(uint8_t) * i,
							cudaMemcpyHostToDevice));
			cout << "attacked String: " << startString << endl;
			if (positionInUsedAlphabet[0] >= usedAlphabetSize)
				break;
			match_found[0] = 0;
			HANDLE_ERROR(
					cudaMemcpy(match_found, dev_match_found, 1,
							cudaMemcpyDeviceToHost));
			if (match_found[0] == 1) {
				break;
			}
			kernel<<<number_of_blocks, threads_per_block, 0, stream2>>>(
					dev_target, dev_result, devPositionInUsedAlphabet,
					iterations_per_thread, dev_match_found, i,
					usedAlphabetSize);
			//change the attacked string for the fourth kernel
			for (int j = i - 1; j > 0; j--) {
				positionInUsedAlphabet[j] = positionInUsedAlphabet[j]
						+ (uint32_t(temp) % usedAlphabetSize);
				temp = temp / usedAlphabetSize;
				if (temp < 1)
					break;
			}
			positionInUsedAlphabet[0] = positionInUsedAlphabet[0]
					+ (uint32_t(temp) % usedAlphabetSize);
			if (temp > usedAlphabetSize) {
				temp = number_of_blocks * threads_per_block
						* iterations_per_thread;
				break;
			}
			for (int j = i - 1; j > 0; j--) {
				while (positionInUsedAlphabet[j] >= usedAlphabetSize) {
					positionInUsedAlphabet[j] -= usedAlphabetSize;
					positionInUsedAlphabet[j - 1]++;
				}
			}
			temp = number_of_blocks * threads_per_block * iterations_per_thread;
			for (int j = 0; j < i; j++) {
				startString[j] = usedAlphabet[positionInUsedAlphabet[j]];
			}
			HANDLE_ERROR(
					cudaMemcpy(devPositionInUsedAlphabet,
							positionInUsedAlphabet, sizeof(uint8_t) * i,
							cudaMemcpyHostToDevice));
			cout << "attacked String: " << startString << endl;
			if (positionInUsedAlphabet[0] >= usedAlphabetSize)
				break;
			match_found[0] = 0;
			HANDLE_ERROR(
					cudaMemcpy(match_found, dev_match_found, 1,
							cudaMemcpyDeviceToHost));
			if (match_found[0] == 1) {
				break;
			}
			kernel<<<number_of_blocks, threads_per_block, 0, stream3>>>(
					dev_target, dev_result, devPositionInUsedAlphabet,
					iterations_per_thread, dev_match_found, i,
					usedAlphabetSize);
			//change the attacked string for the fist kernel
			for (int j = i - 1; j > 0; j--) {
				positionInUsedAlphabet[j] = positionInUsedAlphabet[j]
						+ (uint32_t(temp) % usedAlphabetSize);
				temp = temp / usedAlphabetSize;
				if (temp < 1)
					break;
			}
			positionInUsedAlphabet[0] = positionInUsedAlphabet[0]
					+ (uint32_t(temp) % usedAlphabetSize);
			if (temp > usedAlphabetSize) {
				temp = number_of_blocks * threads_per_block
						* iterations_per_thread;
				break;
			}
			for (int j = i - 1; j > 0; j--) {
				while (positionInUsedAlphabet[j] >= usedAlphabetSize) {
					positionInUsedAlphabet[j] -= usedAlphabetSize;
					positionInUsedAlphabet[j - 1]++;
				}
			}
			for (int j = 0; j < i; j++) {
				startString[j] = usedAlphabet[positionInUsedAlphabet[j]];
			}
			temp = number_of_blocks * threads_per_block * iterations_per_thread;
			if (positionInUsedAlphabet[0] >= usedAlphabetSize)
				break;
			match_found[0] = 0;
			HANDLE_ERROR(
					cudaMemcpy(match_found, dev_match_found, 1,
							cudaMemcpyDeviceToHost));
			if (match_found[0] == 1) {
				break;
			}
		} while (1);
		if (match_found[0] == 1) {
			break;
		}
		/*HANDLE_ERROR( cudaEventRecord( stop_kernel, 0 ) );
		 HANDLE_ERROR( cudaEventSynchronize( stop_kernel ) );
		 HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime_kernel,start_kernel, stop_kernel ) );*/
		//hashes_per_second=(int)((number_of_blocks*threads_per_block*iterations_per_thread)/elapsedTime_kernel*4*1000);
		free(startString);
		free(positionInUsedAlphabet);
	}

	HANDLE_ERROR(cudaStreamSynchronize(stream0));
	HANDLE_ERROR(cudaStreamSynchronize(stream1));
	HANDLE_ERROR(cudaStreamSynchronize(stream2));
	HANDLE_ERROR(cudaStreamSynchronize(stream3));
	HANDLE_ERROR(cudaEventRecord(stop_overall, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop_overall));
	HANDLE_ERROR(
			cudaEventElapsedTime(&elapsedTime_overall, start_overall,
					stop_overall));

	printf("\nTime to generate:  %3.1f ms\n", elapsedTime_overall);
	if (match_found[0] == 1) {
		for (uint16_t i = 0; i <= maxLength; i++) {
			result[i] = 0;
		}
		HANDLE_ERROR(
				cudaMemcpy(result,dev_result,maxLength,cudaMemcpyDeviceToHost));
		cout << result << endl;
	}

	HANDLE_ERROR(cudaEventDestroy(start_kernel));
	HANDLE_ERROR(cudaEventDestroy(stop_kernel));
	HANDLE_ERROR(cudaEventDestroy(start_overall));
	HANDLE_ERROR(cudaEventDestroy(stop_overall));
	cudaFree(dev_usedAlphabet);
	cudaFree(dev_result);
	cudaFree(dev_target);
	cudaFree(dev_match_found);

	HANDLE_ERROR(cudaStreamDestroy(stream0));
	HANDLE_ERROR(cudaStreamDestroy(stream1));
	HANDLE_ERROR(cudaStreamDestroy(stream2));
	HANDLE_ERROR(cudaStreamDestroy(stream3));

	cudaDeviceReset();		//important for profiling
}

int main(int argc, char* argv[]) {
	FILE * fp;
	fp = fopen(argv[1], "r");
	if (fp != NULL) {
		char * line = NULL;
		size_t len = 0;
		uint8_t i;
		char digest[64] = { };
		char result[15] = { };
		uint32_t target[8] = { };

		while ((getline(&line, &len, fp)) != -1) {
			for (i = 0; i < 64; i++)
				digest[i] = line[i];
			for (i = 65; i < strlen(line); i++)
				result[i - 65] = line[i];
			printf("\n========== Expected Result: %s\n", result);
			printf("\n%s\n", digest);
			sha256StringToTarget(digest, target);
			sha256bruteforce(target);
			for (i = 0; i < 64; i++)
				digest[i] = 0;
			for (i = 65; i < strlen(line); i++)
				result[i - 65] = 0;
			for (i = 0; i < 8; i++)
				target[i] = 0;
		}

		if (line)
			free(line);
	} else if (strlen(argv[1]) != 64) {
		// Compute SHA256 digest on cpu
		//CPU Hashing begins here
		char output_cpu[64];
		uint8_t sha256sum_cpu[32];
		sha256comp_cpu(argv[1], strlen(argv[1]), sha256sum_cpu);
		sha256_print(sha256sum_cpu, output_cpu);

//		printf("        Input: %s\n", argv[1]);
//		printf("SHA256-Digest: %s\n", output_cpu);
		printf("%s\n", output_cpu);

	} /*else {
	 /* Unsere targets, wurden bei mir alle gefunden. */
	/*
	 cudaEvent_t start, stop;
	 float time;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaEventRecord(start, 0);
	 cudaThreadSynchronize();

	 uint32_t targets[38][8] = {
	 {0x6a641cbc,0x8ab0602e,0x96c615fb,0x697c6764,0x0e6a1548,0xc1717d0a,0x81cd3bdb,0x64bef43b},  //Durch den bruteforcer hearuasgefunden: deadd
	 {0xcf1e8c45,0x5d16e39f,0xecfd9c20,0x609c46eb,0x9f7566d7,0xc3f7144b,0xe41eb56a,0xc4287826},  //Durch den bruteforcer hearuasgefunden: zxywz
	 {0x3e23e816,0x0039594a,0x33894f65,0x64e1b134,0x8bbd7a00,0x88d42c4a,0xcb73eeae,0xd59c009d},	//b
	 {0xca978112,0xca1bbdca,0xfac231b3,0x9a23dc4d,0xa786eff8,0x147c4e72,0xb9807785,0xafee48bb},	//a
	 {0x34367776,0x2813eaeb,0x65704cc8,0xd9e96f7a,0x444ba0cc,0xa92ff861,0xaf7f6864,0x8b3e6ef1},	//ch
	 {0x1f3ce404,0x15a2081f,0xa3eee75f,0xc39fff8e,0x56c22270,0xd1a978a7,0x249b592d,0xcebd20b4},	//aaaaaaaa
	 {0x38782210,0x12d3785e,0x4f21eef3,0x7119410a,0x7ed8ebb5,0xde28ef82,0xc0cad48d,0x8cdc5d04},	//zzzzzzz	wird nicht gefunden
	 {0x0e10fbe5,0x33c4d8cc,0x73539ce6,0x55057f30,0x92dd975f,0x34aad4f4,0x97f28ad6,0x6ffb503d},	//zzzzzza	wird nicht gefunden
	 {0x7b70d3ab,0x4c764154,0x2e1f158b,0x458eeae7,0xcfb7bdb8,0x15d4110c,0xc6178baf,0xcfdf43f8},	//xxxxxxx
	 {0x67a61945,0x7aae3e86,0x9af3e7c9,0x2078424a,0x773397c1,0x520a9cec,0x76fde54e,0xe8350137},	//qqqqqqq	wird nicht gefunden
	 {0x6ce53691,0xb126808d,0x3745d72b,0x9016384e,0xf0a17400,0x4dc9d3ab,0x151e3c82,0xda186ba5},	//ppppppp	wird nicht gefunden	i>140
	 {0xcc3da533,0x1df88e23,0x2ac0885e,0x142ef9c0,0x27c81432,0x3df9da1c,0xda745b53,0x8f4e950e},	//ooooooo	wird gefunden	i>150
	 {0xbe1f4743,0xf2148891,0x7c4fccfe,0xc99385ee,0xdf039b57,0xb77deba7,0x8b80d5ea,0x04bccb1e},	//nnnnnnn	wird gefunden	i=76
	 {0x800955a7,0xf19d86bd,0x5cca3153,0x050c7299,0xb28f30e9,0x1c7fe854,0xf08a878c,0x158f5aa6},	//lllllll	wird gefunden	i=60
	 {0x5de475c5,0x4f292d35,0x7b4665c4,0xa0667335,0x4d0af583,0xabec2ac5,0x1b752fdf,0x06fcdbbd},	//kkkkkkk	wird gefunden
	 {0x73f5c123,0x3741e5cb,0x12b22c75,0x29f55e46,0xc65b96d7,0xeecce91b,0xc412ca87,0x2c317e45},	//jjjjjjj	wird gefunden
	 {0x15c45977,0xedc54496,0x0301cea5,0xeeb3f6d0,0xac2b96b2,0xd542fd42,0x518ab24d,0xb9c7f829},	//iiiiiii	wird gefunden
	 {0x589f6fec,0xa8b16ba6,0x37fc8a8e,0xa35eea5c,0x224b27e0,0xa65f306e,0x19de14cb,0x0398965a},	//hhhhhhh	wird gefunden
	 {0x85e45110,0x6fb40954,0x4d0edca7,0x0f030bbe,0x905ed7b7,0xd1de93ca,0x20e5390a,0x0a7f3fd5},	//ggggggg	wird gefunden
	 {0x48c1caf3,0x30ea2b39,0xdf2bd04a,0x9b19f344,0x4fd89d61,0xbda17b51,0x1178fd20,0x74b25f9b},	//deadbee	wird gefunden
	 {0x51f36bf2,0x0bc6debb,0x25fe98c5,0x29ecc718,0xc75c2052,0x808521ac,0x4b333a07,0x24881ec5},	//gfedcbb	wird gefunden
	 {0x3de47205,0xe772b39d,0x369b811a,0x8cc515a3,0x1cf31051,0x1bf1f452,0x9b723498,0x8509da55},	//eeeeeee	wird gefunden
	 {0xba24a289,0x0228ef70,0xa16259bb,0x9b72fb1b,0x42ad5f29,0xd7c60a02,0x98456adb,0x4ee11737},	//ddddddd	wird gefunden
	 {0xcd4f3afa,0x6982937e,0x1fcc283c,0x451f1cd9,0xd368e998,0xb554ff7b,0x06eabe6e,0x723a39c2},	//ccccccc	wird gefunden
	 {0xea415a61,0xbd199150,0x84366a0a,0x2fdaebe0,0x70a9c316,0x8877ecdb,0x5e36f490,0x5b5f8aa3},	//bbbbbbb	wird gefunden
	 {0x7d1a5412,0x7b222502,0xf5b79b5f,0xb0803061,0x152a44f9,0x2b37e23c,0x6527baf6,0x65d4da9a},	//abcdefg	wird gefunden
	 {0xcaac75ef,0x1fa69625,0x5f61addf,0x40d7d11d,0x246ed5fe,0xdff2636f,0xbee45e0f,0xe56a1340},	//aaaaaab
	 {0xe4624071,0x4b5db3a2,0x3eee6047,0x9a623efb,0xa4d633d2,0x7fe4f03c,0x904b9e21,0x9a7fbe60},	//aaaaaaa	wird nicht gefunden
	 {0x95fbeb8f,0x769d2c00,0x79d1d113,0x48877da9,0x44aaefab,0xa6ecf9f7,0xf7dab634,0x4ece8605},	//zzzzzz
	 {0xb7fb2176,0x94ae2d30,0x5e766608,0xd250f797,0xdaa984e4,0xac4b5fa6,0x38a729be,0x352f2fcd},	//xxxxxx
	 {0xe2dbf8f5,0xc4cc1514,0x80213d21,0xf95c72aa,0x73a001bc,0xe4915b17,0x691ae409,0x52dcd793},	//ffffff
	 {0x68a55e5b,0x1e43c67f,0x4ef34065,0xa86c4c58,0x3f532ae8,0xe3cda7e3,0x6cc79b61,0x1802ac07},	//zzzzz
	 {0xeaf16bc0,0x7968e013,0xf3f94ab1,0x34247243,0x4a39fc34,0x75f11cf3,0x41a6c396,0x5974f8e9},	//xxxxx
	 {0x99834619,0xb3c16024,0x8b69c7f4,0x2ba868f9,0x45a0ea04,0xcd31cf2f,0x60dc4bc8,0xf7d13b8a},	//fffff
	 {0x2164b17a,0x27fbc64a,0x987a8c8f,0x23341dfd,0x9cef6b90,0x7d75d87b,0x124df41c,0x25812439},	//dzvkb
	 {0x85d0bd50,0xac261c8d,0x3836a816,0x1091b88a,0x12926926,0xcf757e2e,0x2d99c5bd,0x2f452d75},	//deadb
	 {0xba7816bf,0x8f01cfea,0x414140de,0x5dae2223,0xb00361a3,0x96177a9c,0xb410ff61,0xf20015ad}, //abc
	 {0xb8d19b62,0x06b72bf4,0x03a0a87a,0xb21a135a,0x615cc8c5,0x35633000,0x224c831d,0x8a0eb0e3} //dea
	 }; //38 targets

	 uint8_t i,j;
	 uint32_t target[8] = {};
	 for(i = 0; i < 38; i++) {
	 for(j = 0; j < 8; j++) {
	 target[j] = targets[i][j];
	 printf("%08x",target[j]);
	 }
	 printf("\n");
	 sha256bruteforce(target, arguments.verbose);
	 printf("\n\n");
	 }

	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&time, start, stop);

	 cudaEventDestroy(start);
	 cudaEventDestroy(stop);

	 printf("[CPU] took %fms to calcualte.\n", time);

	 /*
	 int i = 2000;
	 while(i-- > 0) {
	 srand(time(NULL ));
	 int someRandValue = rand() % 5;

	 //CPU Hashing begins here
	 char output_cpu[64];
	 uint8_t sha256sum_cpu[32];
	 char* input_string = randstring(7 + someRandValue);
	 sha256comp_cpu(input_string, strlen(input_string), sha256sum_cpu);
	 sha256_print(sha256sum_cpu, output_cpu);

	 printf("\n        Input: %s\n", input_string);
	 printf("SHA256-Digest: %s\n", output_cpu);

	 uint32_t target[8] = { };
	 sha256StringToTarget(output_cpu, target);
	 sha256bruteforce(target, arguments.verbose);
	 }
	 } */
	else {
		uint32_t target[8] = { };
		sha256StringToTarget(argv[1], target);
		sha256bruteforce(target);
	}

	return EXIT_SUCCESS;
}
