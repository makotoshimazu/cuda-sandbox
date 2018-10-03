//
// wmma.cpp
//
// Copyright 2018 Makoto Shimazu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>

#include "kernel.h"

double GFLOPS(double ms, int m, int n, int k) {
  return (2LL * m * n * k / ms) / 1E6;
}

int main() {
  int M = 4096;
  int N = 4096;
  int K = 4096;
  double ms = wmma_run(M, N, K);
  printf("%3.1f ms\n", ms);
  printf("%6.1f GFLOPS\n", GFLOPS(ms, M, N, K));
  return 0;
}
