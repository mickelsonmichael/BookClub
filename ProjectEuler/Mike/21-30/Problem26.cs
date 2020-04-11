using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// A unit fraction contains 1 in the numerator. The decimal representation of the unit fractions with denominators 2 to 10 are given:
	/// 
	/// 1/2 = 0.5
	/// 1/3 = 0.(3)
	/// 1/4 = 0.25
	/// 1/5 = 0.2
	/// 1/6 = 0.1(6)
	/// 1/7 = 0.(142857)
	/// 1/8 = 0.125
	/// 1/9 = 0.(1)
	/// 1/10 = 0.1
	/// 
	/// Where 0.1(6) means 0.166666..., and has a 1-digit recurring cycle. It can be seen that 1/7 has a 6-digit recurring cycle.
	/// 
	/// Find the value of d < 1000 for which 1/d contains the longest recurring cycle in its decimal fraction part.
	/// </summary>
	public static class Problem26
	{
		private const int MaxD = 1000;

		public static void Run()
		{
			var str = GetLongestRepeating();

			Console.WriteLine(str);
		}

		public static int GetLongestRepeating()
		{
			return Enumerable.Range(2, MaxD - 1)
				.Select(d => new { d, repeating = GetRepeating(1, d) })
				.Where(x => !string.IsNullOrWhiteSpace(x.repeating))
				.OrderByDescending(x => x.repeating.Length)
				.First()
				.d;
		}

		public static string GetRepeating(int numerator, int denominator)
		{
			var result = new StringBuilder();
			var cache = new List<int>();

			while (!cache.Contains(numerator))
			{
				cache.Add(numerator);

				int remainder = numerator % denominator;

				if (remainder == 0) return string.Empty;

				if (numerator == 1)
					result.Append("0.");
				else
					result.Append(numerator / denominator);

				numerator = remainder * 10;
			}

			return result.ToString();
		}
	}
}
