using System;
using System.Collections.Generic;
using System.Linq;

namespace ProjectEuler._21_30
{
	/// <summary>
	/// A permutation is an ordered arrangement of objects.
	/// For example, 3124 is one possible permutation of the digits 1, 2, 3 and 4.
	/// If all of the permutations are listed numerically or alphabetically, we call it lexicographic order.
	/// The lexicographic permutations of 0, 1 and 2 are:
	/// 
	/// 012   021   102   120   201   210
	/// 
	/// What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
	/// </summary>
	public static class Problem24
	{
		public static void Run()
		{
			var numbers = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			const int permutationTarget = 1_000_000;

			var test = GetNthPermutation(numbers, permutationTarget - 1);

			Console.WriteLine(test);
		}

		public static string GetNthPermutation(IEnumerable<int> numbers, int n)
		{
			numbers = numbers.OrderBy(x => x);

			decimal permutations = Factorial(numbers.Count() -1);
			int permutationIndex = (int)(n / permutations);
			var newN = n % permutations;
			var thisNumber = numbers.ElementAt(permutationIndex);
			var newNumbers = numbers.Except(new[] { thisNumber });

			if (newNumbers.Count() == 1) return thisNumber.ToString() + newNumbers.ElementAt(0).ToString();

			return thisNumber.ToString() + GetNthPermutation(newNumbers, (int)newN);
		}

		public static int Factorial(int n)
		{
			int total = 1;

			for (int i = n; i > 0; i--)
				total *= i;

			return total;
		}

		/// <summary>
		/// Initial method of getting permutations. Realized there was a better way than brute force
		/// </summary>
		public static IEnumerable<string> GetPermutations(List<int> numbers)
		{
			return GetPermutations(numbers, new List<int>())
				.Select(x => string.Join("", x));
		}

		/// <summary>
		/// Initial method of getting permutations. Realized there was a better way than brute force
		/// </summary>
		public static List<List<int>> GetPermutations(List<int> numbers, List<int> currentPermutation)
		{
			var permutations = new List<List<int>>();
			var numbersRemaining = numbers.Except(currentPermutation);

			foreach (var num in numbersRemaining)
			{
				var thisPermutation = currentPermutation.Union(new[] { num }).ToList();
				var remaining = numbersRemaining.Where(x => x != num).ToList();

				if (numbersRemaining.Count() == 1)
				{
					permutations.Add(thisPermutation);
				}
				else
				{
					permutations.AddRange(GetPermutations(remaining, thisPermutation));
				}
			}

			return permutations;
		}
	}
}
