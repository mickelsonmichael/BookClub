using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// https://projecteuler.net/problem=21
	/// 
	/// Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
	/// If d(a) = b and d(b) = a, where a ≠ b, then a and b are an amicable pair and each of a and b are called amicable numbers.
	/// 
	/// For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. 
	/// The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.
	/// 
	/// Evaluate the sum of all the amicable numbers under 10000.
	/// </summary>
	public static class Problem21
	{
		private static readonly Dictionary<int, int> Cache = new Dictionary<int, int>();

		public static void Run()
		{
			const int maxNumber = 10000;
			var numbers = new List<int>();

			for (int i = 1; i < maxNumber; i++)
			{
				var sum = GetProperDivisorsSum(i);
				var inverse = GetProperDivisorsSum(sum);

				if (sum != inverse && inverse == i)
					numbers.Add(i);
			}


			Console.WriteLine(numbers.Sum());
		}

		public static int GetProperDivisorsSum(int n)
		{
			if (n == 1) return 1;
			if (Cache.ContainsKey(n)) return Cache[n];

			int sum = Enumerable.Range(1, n - 1)
				.Where(x => n % x == 0)
				.Sum();

			Cache.Add(n, sum);

			return sum;
		}
	}
}
