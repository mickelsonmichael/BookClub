using System;

namespace ProjectEuler._01_10
{
	/// <summary>
	/// <para>2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.</para>
	/// <para>What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?</para>
	/// </summary>
	public static class Problem05
	{
		public static void Run()
		{
			int smallest = MinDivider(1, 20);

			Console.WriteLine(smallest);
		}

		private static int MinDivider(int lowerBound, int upperBound)
		{
			for (int number = upperBound; number < int.MaxValue; number += upperBound)
			{
				for (int factor = lowerBound; factor <= upperBound; factor++)
				{
					if (number % factor != 0) break;

					if (factor == upperBound) return number;
				}
			}

			return int.MaxValue;
		}
	}
}
