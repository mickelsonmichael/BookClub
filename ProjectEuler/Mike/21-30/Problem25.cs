using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// What is the index of the first term in the Fibonacci sequence to contain 1000 digits?
	/// </summary>
	public static class Problem25
	{
		public static void Run()
		{
			var firstSequenceWithThousandDigits = Fib()
				.Select((x, y) => new { Number = x, Digits = GetNumberOfDigits(x), Index = y + 1 })
				.First(x => x.Digits >= 1000);

			Console.WriteLine(firstSequenceWithThousandDigits.Index);
		}

		public static IEnumerable<BigInteger> Fib()
		{
			BigInteger current = 1;
			BigInteger next = 1;

			while (true)
			{
				yield return current;

				BigInteger temp = current;
				current = next;
				next = temp + current;
			}
		}

		private static int GetNumberOfDigits(BigInteger n) => n.ToString().Length;

		/// <summary>
		/// https://blog.dreamshire.com/project-euler-25-solution/
		/// </summary>
		public static int AlternativeSolution(int numberOfDigits)
		{
			double phi = (1 + Math.Sqrt(5)) / 2; // this is the equation for the "golden ratio" which is closely linked to Fibonacci
			// 1000 digits can be represented as 10^999 (or 10^(d-1)
			// the equation to get the nth Fibonacci is F_n = [phi^n / sqrt(5)] for n >= 0
			// Thus for 1000 digits we can say F_n >= 10^999
			// 10^999 < ( phi^n / sqrt(5) )
			// phi^n > sqrt(5) * 10 ^ 999
			// n * log(phi) > log(5)/2 + 999 * log(10)
			// NOTE: sqrt(5) == 5^1/2 and log10(10) == 1
			// n > ( log(5)/2 + 999 ) / log(phi)
			// this results in the equation below. Taking the ceiling of that equation gives the first number that satisfies the condition

			return (int)Math.Ceiling(((Math.Log10(5) / 2) + numberOfDigits - 1) / Math.Log10(phi));
		}
	}
}
