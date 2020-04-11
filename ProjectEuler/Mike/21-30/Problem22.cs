using System;
using System.IO;
using System.Linq;

namespace ProjectEuler._21_30
{
	public static class Problem22
	{
		private static readonly char[] Letters = new[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };

		public static void Run()
		{
			var nameScores = ReadFile()
				.Split(",")
				.Select(x => x.Trim('"').ToLower())
				.OrderBy(x => x)
				.Select((x, y) => new { nameScore = GetNameScore(x), index = y + 1 })
				.Sum(x => x.nameScore * x.index);

			Console.WriteLine(nameScores);
		}

		private static string ReadFile()
		{
			var fileName = Path.Combine("21-30", "p022_names.txt");
			return File.ReadAllText(fileName);
		}

		private static int GetNameScore(string name) => name.ToCharArray().Sum(GetLetterScore);

		private static int GetLetterScore(char letter) => Array.IndexOf(Letters, letter) + 1;
	}
}
