using System;
using System.Linq;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// The sum of the squares of the first ten natural numbers is 385
	/// 1^2 + 2^2 + 3^2 ... 10^2 = 385
	/// The square of the sum of the first ten natural numbers is 3025
	/// (1 + 2 + 3 + ... + 10)^2 = 3025
	/// Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 
	/// 3025 − 385 = 2640
	/// Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
	/// </summary>
	public static class Problem06
	{
		public static void Run()
		{
			const int number = 100;

			var sumOfSquares = SumOfSquares(number);
			var squarOfSum = SquareOfSum(number);
			var difference = squarOfSum - sumOfSquares;

			Console.WriteLine(difference);
		}

		private static long SquareOfSum(int maxNumber)
		{
			var sum = Enumerable.Range(1, maxNumber)
				.Sum();

			return sum * sum;
		}

		private static long SumOfSquares(int maxNumber)
		{
			return Enumerable.Range(1, maxNumber)
				.Sum(x => x * x);
		}
	}
}
