using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._01_10
{
	public static class Problem03
	{
		/// <summary>
		/// <para>The prime factors of 13195 are 5, 7, 13 and 29.</para>
		/// <para>What is the largest prime factor of the number 600851475143 ?</para>
		/// </summary>
		public static void Run()
		{
			Console.WriteLine(nameof(Problem03));

			const long target = 600_851_475_143;

			var results = GetMaxPrime(target);

			Console.WriteLine(results);
		}

		private static long GetMaxPrime(long number)
		{
			// the square of the square root of a number is the number
			// thus, we can limit the search to any numbers LESS than the square root
			// the result of this will be the compliment of the largest
			long start = (long)Math.Floor(Math.Sqrt(number));
			var factors = new List<long>();

			for (var i = 2; i < start; i++)
			{
				if (number % i == 0) // if a factor
				{
					factors.Add(i);
					factors.Add(number / i); // the inverse then is also a factor
				}
			}

			// in most scenarios we could just consider the inverses
			// but we also must consider it's possible that none of the inverses
			// are primes
			var max = factors.OrderByDescending(x => x)
				.First(IsPrime);

			Console.WriteLine(string.Join(", ", factors.ToArray()));

			return max;
		}

		private static bool IsPrime(long number)
		{
			if (number <= 1) return false; // exclude 1 and negative numbers
			if (number == 2) return true; // 2 is always prime
			if (number % 2 == 0) return false; //exclude even numbers

			var limit = Math.Floor(Math.Sqrt(number)); // start with the lower half of the compliments

			for (var i = 3; i <= limit; i+=2)
			{
				if (number % i == 0) return false; // any valid compliment means it is not prime
			}

			return true;
		}
	}
}
