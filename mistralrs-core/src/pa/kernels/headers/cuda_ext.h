#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"

typedef enum {
  DATA_U8 = 0,
  DATA_F16,
  DATA_BF16,
  DATA_F32,
  DATA_F64,
} ScalarType;