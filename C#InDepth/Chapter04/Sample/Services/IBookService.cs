using System.Threading.Tasks;
using Sample.Business;

namespace Sample.Services
{
    public interface IBookService
    {
        Task<Book> GetBookAsync();
    }
}