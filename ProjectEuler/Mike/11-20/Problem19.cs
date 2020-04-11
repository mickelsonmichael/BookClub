using System;

namespace ProjectEuler._11_20
{
	/// <summary>
	/// You are given the following information, but you may prefer to do some research for yourself.
	/// 
	/// 1 Jan 1900 was a Monday.
	/// 
	/// Thirty days has September,
	/// April, June and November.
	/// All the rest have thirty-one,
	/// Saving February alone,
	/// Which has twenty-eight, rain or shine.
	/// And on leap years, twenty-nine.
	/// 
	/// A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
	/// 
	/// How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
	/// </summary>
	public static class Problem19
	{
		private static readonly DateTime FirstMonday = new DateTime(1900, 01, 01);

		public static void Run()
		{
			var start = new DateTime(1901, 01, 01);
			var end = new DateTime(2000, 12, 31);
			int mondays = 0;
			var firstSunday = FirstMonday.AddDays(6);

			for (DateTime day = firstSunday; day <= end; day += TimeSpan.FromDays(7))
			{
				if (day < start) continue;

				if (day.DayOfWeek == DayOfWeek.Sunday && day.Day == 1) mondays++;
			}

			Console.WriteLine(mondays);
		}
	}
}
