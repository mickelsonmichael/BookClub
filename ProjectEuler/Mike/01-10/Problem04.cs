using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// A palindromic number reads the same both ways.The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
	/// Find the largest palindrome made from the product of two 3-digit numbers.
	/// </summary>
	public static class Problem04
	{
		private const int Max = 999;
		private const int Min = 100;

		public static void Run()
		{
			var results = new List<int>();

			for (int i = Max; i >= Min; i--)
			{
				for (int j = Max; j >= Min; j--)
				{
					if (IsPallindromic(i * j))
					{
						results.Add(i * j);
					}
				}
			}

			Console.WriteLine(results.Max());
		}

		private static bool IsPallindromic(long number)
		{
			var str = number.ToString();
			var halfIndex = str.Length / 2;

			var firstHalf = str.Substring(0, halfIndex).ToCharArray();
			var secondHalf = str.Substring(halfIndex).ToCharArray();

			Array.Reverse(secondHalf);

			return firstHalf.SequenceEqual(secondHalf);
		}
	}
}
