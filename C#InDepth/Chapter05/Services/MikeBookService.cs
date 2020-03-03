

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Chapter05.Business;
using Chapter05.Repo;

namespace Chapter05.Services
{
    public class MikeBookService : IBookService
    {
        private IFileReader FileReader = new FileReader();
        private IPlaceholderRepo PlaceholderRepo = new PlaceholderRepo();

        public async Task<Book> GetBookAsync()
        {
            var booksTask = FileReader.GetBooksAsync()
                .ConfigureAwait(false);

            Console.WriteLine("Finding book...");

            return (await booksTask)
                .Single(x => x.Author == "Jon Skeet");
        }

        public async Task<Comment> GetCommentAsync(int id)
        {
            var comments = PlaceholderRepo.GetCommentsAsync();

            Console.WriteLine("Finding comment...");

            return (await comments)
                .Single(x => x.id == id);
        }

        public async Task GetAllAsync()
        {
            var tasks = new List<Task>();

            Console.WriteLine("Start getting books");
            tasks.Add(FileReader.GetBooksAsync());
            Console.WriteLine("Start getting comments");
            tasks.Add(PlaceholderRepo.GetCommentsAsync());

            await Task.WhenAll(tasks)
                .ContinueWith((task) => Console.WriteLine("All done"));
        }
    }
}