using System;
using System.Linq;
using System.Numerics;

namespace ProjectEuler._11_20
{
	public static class Problem16
	{
		public static void Run()
		{
			const int exponent = 1000;

			var sum = BigInteger.Pow(2, exponent)
				.ToString()
				.ToCharArray()
				.Select(x => char.GetNumericValue(x))
				.Sum();

			Console.WriteLine(sum);
		}
	}
}
