using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._01_10
{
	public static class Problem02
	{
		public static void Calculate()
		{
			var sum = GetFibbonachi()
				.Sum();

			Console.WriteLine(sum);
		}

		private static IEnumerable<int> GetFibbonachi()
		{
			var currentValue = 1;
			var lastValue = 0;

			while (currentValue <= 4_000_000)
			{
				int newValue = currentValue + lastValue;

				lastValue = currentValue;
				currentValue = newValue;

				if (currentValue % 2 != 0) continue;

				yield return currentValue;
			}
		}
	}
}
