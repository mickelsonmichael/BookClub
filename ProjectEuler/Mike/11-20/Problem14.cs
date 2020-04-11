using System;
using System.Linq;

namespace ProjectEuler._11_20
{
	/// <summary>
	/// <para>
	/// The following iterative sequence is defined for the set of positive integers:</para>
	/// <para>
	/// n -> n/2 (n is even)
	/// n -> 3n + 1 (n is odd)
	/// </para>
	/// <para>
	/// Using the rule above and starting with 13, we generate the following sequence:
	/// 13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 - 1
	/// </para>
	/// <para>
	/// It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. 
	/// Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
	/// </para>
	/// <para>Which starting number, under one million, produces the longest chain?</para>
	/// <para>NOTE: Once the chain starts the terms are allowed to go above one million.</para>
	/// </summary>
	public static class Problem14
	{
		public static void Run()
		{
			const int maxStartingNumber = 1_000_000;

			var maxChainNumber = Enumerable.Range(1, maxStartingNumber)
				.Select(x => new { n = x, Length = GetChainLength(x) })
				.OrderByDescending(x => x.Length)
				.First();

			Console.WriteLine(maxChainNumber.n);
		}

		private static int GetChainLength(long n)
		{
			int chainLength = 1;

			while (n > 1)
			{
				if (IsEven(n))
					n = DoEven(n);
				else
					n = DoOdd(n);

				chainLength++;
			}

			return chainLength;
		}

		private static long DoOdd(long n) => (3 * n) + 1;
		private static long DoEven(long n) => n / 2;
		private static bool IsEven(long n) => n % 2 == 0;
	}
}
