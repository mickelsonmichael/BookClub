using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._11_20
{
	/// <summary>
	/// If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
	/// 
	/// If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used? 
	/// 
	/// NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. 
	/// The use of "and" when writing out numbers is in compliance with British usage.
	/// </summary>
	public static class Problem17
	{
		private static readonly Dictionary<int, string> Ones = new Dictionary<int, string>
		{
			{ 1, "one" },
			{ 2, "two" },
			{ 3, "three" },
			{ 4, "four" },
			{ 5, "five" },
			{ 6, "six" },
			{ 7, "seven" },
			{ 8, "eight" },
			{ 9, "nine" }
		};

		private static readonly Dictionary<int, string> Tens = new Dictionary<int, string>
		{
			{ 10, "ten" },
			{ 11, "eleven" },
			{ 12, "twelve" },
			{ 13, "thirteen" },
			{ 14, "fourteen" },
			{ 15, "fifteen" },
			{ 16, "sixteen" },
			{ 17, "seventeen" },
			{ 18, "eighteen" },
			{ 19, "nineteen" }
		};

		public static void Run()
		{
			const int number = 1000;
			var exclude = new char[] { ' ', '-', '\t', '\n' };

			var numberOfLetters = Enumerable.Range(1, number)
				.Select(GetText)
				.SelectMany(x => x.ToCharArray())
				.Count(x => !exclude.Contains(x));

			Console.WriteLine(numberOfLetters);
		}

		public static string GetText(int number)
		{
			if (number < 10) return Ones[number];
			if (number < 20) return Tens[number];
			if (number < 100) return GetTens(number);
			if (number < 1000) return GetHundreds(number);
			else return "One thousand";
		}

		private static string GetTens(int number)
		{
			var n = (int)Math.Floor(number / 10d);
			var str = string.Empty;

			if (n == 2) str = "twenty";
			else if (n == 3) str = "thirty";
			else if (n == 4) str = "forty";
			else if (n == 5) str = "fifty";
			else if (n == 6) str = "sixty";
			else if (n == 7) str = "seventy";
			else if (n == 8) str = "eighty";
			else if (n == 9) str = "ninety";

			if (number - (n * 10) > 0)
				str += $"-{Ones[number - (n*10)]}";

			return str;
		}

		private static string GetHundreds(int number)
		{
			var n = (int)Math.Floor(number / 100d);
			var str = $"{Ones[n]} hundred";

			var tens = number - (n * 100);
			if (tens > 0)
			{
				str += " and ";

				if (tens < 10) str += Ones[tens];
				else if (tens < 20) str += Tens[tens];
				else str += GetTens(number - (n * 100));
			}

			return str;
		}
	}
}
