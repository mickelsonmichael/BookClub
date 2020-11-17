using System;
using System.Linq;
using DapperTrial.Repo;

namespace DapperTrial
{
    public static class Program
    {
        public static void Main()
        {
            Console.WriteLine("Getting repo...");
            var repo = new BookRepository();

            var books = repo.GetBooks();

            Console.WriteLine("Books: {0}", books.Count());

            repo.Cleanup();
        }
    }
}
