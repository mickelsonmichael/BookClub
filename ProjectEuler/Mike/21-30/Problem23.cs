using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. 
	/// For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.
	/// 
	/// A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.
	/// 
	/// As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. 
	/// By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. 
	/// However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number 
	/// that cannot be expressed as the sum of two abundant numbers is less than this limit.
	/// 
	/// Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers
	/// </summary>
	public static class Problem23
	{
		private const int Max = 28123;
		private const int Min = 12;

		public static void Run()
		{
			var numbers = GetNumbers();

			Console.WriteLine(numbers.Sum());
		}

		public static IEnumerable<int> GetNumbers()
		{
			var NumbersRemaining = Enumerable.Range(1, Max - 1).ToHashSet();
			var abundantNumbers = GetAbundantNumbers().ToList();

			foreach (var number in abundantNumbers)
			{
				foreach (var number2 in abundantNumbers)
				{
					NumbersRemaining.Remove(number + number2);
				}
			}

			return NumbersRemaining;
		}

		public static IEnumerable<int> GetAbundantNumbers()
		{
			for (int i = Min; i < Max; i++)
			{
				var sumOfFactors = GetFactors(i)
					.Sum();

				if (sumOfFactors > i) yield return i;
			}
		}

		public static IEnumerable<int> GetFactors(int n)
		{
			var upperLimit = Math.Floor(Math.Sqrt(n));

			for (int i = 1; i <= upperLimit; i++)
			{
				if (n % i == 0)
				{
					yield return i;
					if (i == 1) continue;
					if (i == n / i) continue;
					yield return n / i;
				}
			}
		}
	}
}
