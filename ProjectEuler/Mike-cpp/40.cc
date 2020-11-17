#include <iostream>
#include <benchmark/benchmark.h>

using namespace std;

void Solve() {
  string s;
  
  size_t max = 1000000;
  size_t n = 1, result = 1, count = 0;
  size_t dec = 1;

  while(n <= max) {
    s = to_string(n);

    for (char c : s) {
      count += 1;
      
      if (count == dec) {
        int val = c - '0';
        result *= val;
        dec *= 10;	
      }
    }

    n++;
  } 

 // cout << result << '\n';
}

static void Benchmark_Solve(benchmark::State &state) {
  for (auto _ : state)
	  Solve();
}

BENCHMARK(Benchmark_Solve)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

