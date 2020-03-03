using System;
using System.Threading.Tasks;
using Chapter05.Services;

namespace Chapter05
{
    class Program
    {
        private static MikeBookService Service = new MikeBookService();

        async static Task Main(string[] args)
        {
            await DemoGetBook();
            await DemoGetComments();
            await DemoParallel();

            await ValueTaskDemo.Demo();
        }

        private static async Task DemoGetBook()
        {
            var book = await Service.GetBookAsync();

            Console.WriteLine("Got Book");
            Console.WriteLine(book.Title);

            Console.WriteLine("-----");
        }

        private static async Task DemoGetComments()
        {
            var comment = await Service.GetCommentAsync(10);
            Console.WriteLine("Got comment");
            Console.WriteLine(comment.body);
            
            Console.WriteLine("-----");
        }

        private static async Task DemoParallel()
        {
            await Service.GetAllAsync();
            Console.WriteLine("-----");
        }
    }
}
