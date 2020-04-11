using System;
using System.Linq;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
	/// 
	/// Find the sum of all the multiples of 3 or 5 below 1000.
	/// </summary>
	public static class Problem01
	{
		public static bool Calculate(int maxNumber)
		{
			var sum = Enumerable.Range(1, maxNumber - 1)
				.Where(IsMultipleOfThreeOrFive)
				.Sum();

			Console.WriteLine(sum);

			return true;
		}

		private static bool IsMultipleOfThreeOrFive(int number)
		{
			return number % 3 == 0 || number % 5 == 0;
		}
	}
}
