using System;
using System.Threading.Tasks;
using Sample.Services;

namespace Sample
{
    class Program
    {
        async static Task Main(string[] args)
        {
            var service = new MikeBookService();

            var book = await service.GetBookAsync();

            Console.WriteLine("Got Book");
            Console.WriteLine(book.Title);
        }
    }
}
