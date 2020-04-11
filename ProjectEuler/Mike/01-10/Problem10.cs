using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
	/// Find the sum of all the primes below two million.
	/// </summary>
	public static class Problem10
	{
		public static void Run()
		{
			const long maxPrime = 2_000_000;
			var primes = GetPrimes(maxPrime);
			var sum = primes.Sum();

			Console.WriteLine(sum);
		}

		private static IEnumerable<long> GetPrimes(long maximum)
		{
			return Enumerable.Range(1, (int)maximum)
				.Where(Utilities.IsPrime)
				.Select(n => (long)n);
		}
	}
}
