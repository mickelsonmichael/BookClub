using Sample.Business;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Sample.Repo {
    public interface IFileReader
    {
        Task<IEnumerable<Book>> GetBooksAsync();
    }
}