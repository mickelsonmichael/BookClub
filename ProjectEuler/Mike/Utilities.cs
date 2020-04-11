using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace ProjectEuler
{
	public static class Utilities
	{
		public static bool IsPrime<T>(this T value) where T : struct
		{
			long val = Convert.ToInt64(value);

			if (val <= 1) return false; // exclude 1 and negative numbers
			if (val == 2) return true; // 2 is always prime
			if (val % 2 == 0) return false; //exclude even numbers

			var limit = Math.Floor(Math.Sqrt(val)); // start with the lower half of the compliments

			for (var i = 3; i <= limit; i += 2)
			{
				if (val % i == 0) return false; // any valid compliment means it is not prime
			}

			return true;
		}

		private readonly static Stopwatch Stopwatch = new Stopwatch();

		public static void Time(Action problem)
		{
			var cancellationSource = new CancellationTokenSource();

			Console.WriteLine("Starting Problem Calculation...");
			Task.Run(Benchmark, cancellationSource.Token);

			Stopwatch.Start();

			problem();

			Stopwatch.Stop();
			cancellationSource.Cancel();

			var totalTime = Stopwatch.Elapsed.TotalSeconds;

			Console.WriteLine($"Total Time: {totalTime}");

			if (totalTime > 60) Console.WriteLine("TIME WAS OVER THE 1 MINUTE SUGGESTION. MAY NEED REFACTORING");
		}

		private static void Benchmark()
		{
			while (true)
			{
				Thread.Sleep(TimeSpan.FromSeconds(10));

				Console.WriteLine($"{(int)Stopwatch.Elapsed.TotalSeconds}s");
			}
		}
	}
}
