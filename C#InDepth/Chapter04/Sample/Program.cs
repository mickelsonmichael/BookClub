using System;
using System.Linq;
using System.Threading.Tasks;
using Sample.Repo;

namespace Sample
{
    class Program
    {
        async static Task Main(string[] args)
        {
            var repo = new FileReader();

            var books = await repo.GetBooksAsync();

            Console.WriteLine("Got Books");
            Console.WriteLine(books.Count());
        }
    }
}
