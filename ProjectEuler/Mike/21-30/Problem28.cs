using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace ProjectEuler._21_30
{
	public static class Problem28
	{
		public static void Run()
		{
			var grid = MakeGrid(5, 5);

			Print(grid);
		}

		private static int[,] MakeGrid(int gridX, int gridY)
		{
			if (gridX != gridY || gridX % 2 == 0) throw new InvalidOperationException("Grid must have odd numbered, equivalent dimensions (e.g. 5x5 or 101x101)");

			var grid = new int[gridX,gridY];
			int half = (int)Math.Floor(gridX / 2d);
			int currentSize = 1;
			int n = 1;
			int currentX = half, currentY = half;
			(int x, int y) pointer = (half, half);

			while (currentSize <= gridX)
			{
				var min = half - (int)Math.Floor(currentSize / 2d);
				var max = half + (int)Math.Floor(currentSize / 2d);

				while (pointer.y < max)
				{
					grid[pointer.y, pointer.x] = n++;
					pointer.y++;
				}

				while (pointer.x < max)
				{
					grid[pointer.y, pointer.x] = n++;
					pointer.x++;
				}

				while (pointer.y > min)
				{
					grid[pointer.y, pointer.x] = n++;
					pointer.y--;
				}

				while (pointer.x > min)
				{
					grid[pointer.y, pointer.x] = n++;
					pointer.x--;
				}

				


				pointer.y++;
				currentSize += 2;
			}

			return grid;
		}

		private static void Print(int[,] grid)
		{
			for (int i = 0; i < grid.GetLength(0); i++)
			{
				for (int j = 0; j < grid.GetLength(1); j++)
				{
					Console.Write($"{grid[i, j].ToString("00")} ");
				}
				Console.Write("\n");
			}
		}
	}
}
