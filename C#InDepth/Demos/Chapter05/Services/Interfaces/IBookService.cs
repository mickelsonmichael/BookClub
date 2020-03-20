using System.Threading.Tasks;
using Chapter05.Business;

namespace Chapter05.Services
{
    public interface IBookService
    {
        // Run both of the other methods and write to the console when complete
        Task GetAllAsync();
        // Retrieve one book (can be any book) from the FileReader Repo
        Task<Book> GetBookAsync();
        // Return a comment from the PlaceholderRepo
        Task<Comment> GetCommentAsync(int id);
    }
}