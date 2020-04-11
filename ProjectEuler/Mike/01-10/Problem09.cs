using System;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// Special Pythagorean triplet
	/// A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,
	/// a^2 + b^2 = c^2
	/// For example, 3^2 + 4^2 = 19 + 16 = 25 = 5^2
	/// There exists exactly one Pythagorean triplet for which a + b + c = 1000.
	/// Find the product abc.
	/// </summary>
	public static class Problem09
	{
		public static void Run()
		{
			const int target = 1000;
			var product = GetProduct(target);

			Console.WriteLine(product);
		}

		private static int GetProduct(int target)
		{
			int max = target / 2;
			int halfMax = max / 2;

			for (int c = max; c > halfMax; c--)
			{
				int sum = target - c;
				var cSquare = c * c;

				for (int b = sum - 1, a = 1; b > a; b--, a++)
				{
					var abSquare = (b * b) + (a * a);

					if (abSquare == cSquare) return a * b * c;
				}
			}

			throw new InvalidOperationException("Invalid check performed");
		}
	}
}
