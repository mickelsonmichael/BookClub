using System;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
	/// What is the 10,001st prime number?
	/// </summary>
	public static class Problem07
	{
		public static void Run()
		{
			const int nth = 10_001;
			var prime = FindNthPrime(nth);

			Console.WriteLine(prime);
		}

		private static int FindNthPrime(int n)
		{
			int primeCount = 0;

			for (int i = 2; i < int.MaxValue; i++)
			{
				if (IsPrime(i))
				{
					primeCount++;

					if (primeCount == n) return i;
				}
			}

			return 2;
		}

		private static bool IsPrime(long number)
		{
			if (number <= 1) return false; // exclude 1 and negative numbers
			if (number == 2) return true; // 2 is always prime
			if (number % 2 == 0) return false; //exclude even numbers

			var limit = Math.Floor(Math.Sqrt(number)); // start with the lower half of the compliments

			for (var i = 3; i <= limit; i += 2)
			{
				if (number % i == 0) return false; // any valid compliment means it is not prime
			}

			return true;
		}
	}
}
