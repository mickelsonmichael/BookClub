using System;
using System.Threading.Tasks;
using Chapter05.Services;

namespace Chapter05
{
    class Program
    {
        async static Task Main(string[] args)
        {
            var service = new MikeBookService();

            var book = await service.GetBookAsync();

            Console.WriteLine("Got Book");
            Console.WriteLine(book.Title);

            Console.WriteLine("-----");

            var comment = await service.GetCommentAsync(10);
            Console.WriteLine("Got comment");
            Console.WriteLine(comment.body);
            
            Console.WriteLine("-----");

            await service.GetAllAsync();
        }
    }
}
