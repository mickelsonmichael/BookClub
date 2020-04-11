using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// Euler discovered the remarkable quadratic formula:
	/// n^2 + n + 41
	/// 
	/// It turns out that the formula will produce 40 primes for the consecutive integer values 0≤n≤39. 
	/// However, when n=40,402+40+41=40(40+1)+41 is divisible by 41, and certainly when n=41,412+41+41 is clearly divisible by 41.
	/// 
	/// The incredible formula n^2−79n+1601 was discovered, which produces 80 primes for the consecutive values 0≤n≤79. 
	/// The product of the coefficients, −79 and 1601, is −126479.
	/// 
	/// Considering quadratics of the form:
	///n^2 + an + b, where |a| < 1000 and |b| <= 1000
	///where |n| is the modulus/absolute value of n
	///e.g. |11| = 11 and |-4| = 4
	///
	/// Find the product of the coefficients, a and b, for the quadratic expression that produces the maximum number of primes for consecutive values of n, starting with n=0.
	/// </summary>
	public static class Problem27
    {
		private static readonly Dictionary<long, bool> PrimeCache = new Dictionary<long, bool>();

		public static void Run()
        {
			long n = GetLargestProductOfEulerCoefficients();

			Console.WriteLine(n);
		}

		public static long GetLargestProductOfEulerCoefficients()
		{
			int longestProductChain = 0;
			long largestProductOfCoefficients = 0;

			for (int a = -999; a < 1000; a++)
			{

				for (int b = -1000; b <= 1000; b++)
				{
					int n = 0;
					int primeCount = 0;

					while (true)
					{
						long result = (n * n) + (a * n) + b;

						if (!IsPrime(result)) break;

						primeCount++;
						n++;
					}

					long product = a * b;

					if (primeCount > longestProductChain)
					{
						longestProductChain = primeCount;
						largestProductOfCoefficients = product;
					}
				}
			}

			return largestProductOfCoefficients;
		}

		public static bool IsPrime(long n)
		{
			if (!PrimeCache.ContainsKey(n))
				PrimeCache.Add(n, n.IsPrime());

			return PrimeCache[n];
		}
	}
}
