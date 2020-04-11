using System;
using System.Collections.Generic;

namespace ProjectEuler._11_20
{
	public static class Problem15
	{
		private static Dictionary<(int, int), long> Cache;

		public static void Run()
		{
			var numRoutes = FindNumberOfRoutes();

			Console.WriteLine(numRoutes.ToString("n0"));
		}

		public static long FindNumberOfRoutes()
		{
			Cache = new Dictionary<(int, int), long>();

			return FindNumberOfRoutes((0, 0));
		}

		public static long FindNumberOfRoutes((int x, int y) coords)
		{
			const int MaxX = 20;
			const int MaxY = 20;
			long numberOfRoutes = 0;

			if (Cache.ContainsKey(coords))
				return Cache[coords];

			if (coords.x < MaxX)
				numberOfRoutes += FindNumberOfRoutes((coords.x + 1, coords.y));

			if (coords.y < MaxY)
				numberOfRoutes += FindNumberOfRoutes((coords.x, coords.y + 1));

			if (coords.y == MaxY && coords.x == MaxX)
				return 1;

			Cache[coords] = numberOfRoutes;

			return numberOfRoutes;
		}
	}
}
