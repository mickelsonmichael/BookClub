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
        private ISampleFileReader FileReader = new SampleFileReader();
        private IPlaceholderRepo PlaceholderRepo = new PlaceholderRepo();

        public async Task<Book> GetBookAsync()
        {
            // start the task now, but we don't use the result yet
            var booksTask = FileReader.GetBooksAsync()
                .ConfigureAwait(false); // don't capture the context

            Console.WriteLine("Finding book...");
            // this could be some more logic that we can do before we get the book back

            return (await booksTask) // wait for the task to finish
                .Single(x => x.Author == "Jon Skeet"); // use the result like nomral
        }

        public async Task<Comment> GetCommentAsync(int id)
        {
            // start the task now
            var comments = PlaceholderRepo.GetCommentsAsync()
                .ConfigureAwait(false);
            
            // if we wanted to, we could just await the results right away
            // var comments = await PlaceholderRepo.GetCommentsAsync();
            // we then wouldn't need to await it later since we'd have the results

            Console.WriteLine("Finding comment...");

            return (await comments)
                .Single(x => x.id == id);
        }

        // this demonstrates how two tasks can be run in parallel
        // by starting both tasks then waiting for the results
        public async Task GetAllAsync()
        {
            // start the GetBooksAsync task running
            Console.WriteLine("Start getting books in parallel");
            var bookTask = FileReader.GetBooksAsync();

            // start the GetCommentsAsync task running
            Console.WriteLine("Start getting comments in parallel");
            var commentsTask = PlaceholderRepo.GetCommentsAsync();

            // create a list of tasks to do
            var tasks = new Task[] { bookTask, commentsTask }; 

            await Task.WhenAll(tasks) // when all the tasks have finished
                .ContinueWith((task) => Console.WriteLine("All done")); // do this thing (anonymous method, lambda, or any other method with a matching signature)
        }
    }
}