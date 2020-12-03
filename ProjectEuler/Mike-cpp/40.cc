#include <iostream>
#include <benchmark/benchmark.h>

using namespace std;

// Brute force method
void Solve() {
  const unsigned int max = 1000000;
  unsigned int result = 1, digit_pos = 0, target = 1;

  for (unsigned int i = 1; i <= max; i++) {
    string iStr = to_string(i); // convert the number to a string

    // increment over each digit in the number
    for (char c : iStr) {
      digit_pos += 1;
     
      // if we're at the target position, grab the digit
      if (digit_pos == target) {
        int val = c - '0';
        result *= val;
        target *= 10;	
      }
    }
  } 

  // Output result to standard output
	// cout << result << '\n';
}

static void Benchmark_Solve(benchmark::State &state) {
  for (auto _ : state)
	  Solve();
}

BENCHMARK(Benchmark_Solve)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

