#ifndef _SHA256CPU_H
#define _SHA256CPU_H

// Standard includes
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
	uint32_t total[2];
	uint32_t state[8];
	uint8_t buffer[64];
} sha256_context;

/* Used by main to communicate with parse_opt. */
typedef struct arguments {
	char *args[1]; /* arg1 */
	int sha256, benchmark, verbose;
} arguments;

#ifdef __cplusplus
extern "C"
#endif
void sha256comp_cpu(char* input, uint32_t input_length, uint8_t* sha256sum);
#ifdef __cplusplus
extern "C"
#endif
void sha256_print(unsigned char sha256sum[32], char output[64]);
#ifdef __cplusplus
extern "C"
#endif
void sha256StringToTarget(char input[64], uint32_t output[8]);
#ifdef __cplusplus
extern "C"
#endif
char *randstring(size_t length);
#ifdef __cplusplus
extern "C"
#endif
int readFileLineByLine(char* filePath, char* output);


#endif /* sha256cpu.h */