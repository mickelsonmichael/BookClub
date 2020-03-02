

using System;
using System.Linq;
using System.Threading.Tasks;
using Sample.Business;
using Sample.Repo;

namespace Sample.Services
{
    public class MikeBookService : IBookService
    {
        private IFileReader FileReader = new FileReader();

        public async Task<Book> GetBookAsync()
        {
            var booksTask = FileReader.GetBooksAsync();

            Console.WriteLine("Finding book...");

            return (await booksTask)
                .Single(x => x.Author == "Jon Skeet");
        }
    }
}