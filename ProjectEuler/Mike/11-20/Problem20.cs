using System;
using System.Linq;
using System.Numerics;

namespace ProjectEuler._11_20
{
	/// <summary>
	/// n! means n × (n − 1) × ... × 3 × 2 × 1
	/// 
	/// For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
	/// and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.
	/// 
	/// Find the sum of the digits in the number 100!
	/// </summary>
	public static class Problem20
	{
		public static void Run()
		{
			const int number = 100;

			var result = GetFactorial(number)
				.ToString()
				.ToCharArray()
				.Sum(x => char.GetNumericValue(x));

			Console.WriteLine(result);
		}

		public static BigInteger GetFactorial(int number)
		{
			var factorial = (BigInteger)number;

			for (int n = number - 1; n > 0; n--)
			{
				factorial *= n;
			}

			return factorial;
		}
	}
}
