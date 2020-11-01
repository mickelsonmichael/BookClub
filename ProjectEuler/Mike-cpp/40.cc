#include <iostream>

using namespace std;

int main() {
  string s;
  size_t n = 1, result = 1, count = 0;

  while(n <= 1000000) {
    s = to_string(n);

    for (char c : s) {
      count += 1;
      switch (count) {
        case 1:
	case 10:
	case 100:
	case 1000:
	case 10000:
	case 100000:
	case 1000000:
	  int val = c - '0';
          result *= val;
      } 
    }

    n++;
  } 

  cout << result << '\n';

  return 0;
}

