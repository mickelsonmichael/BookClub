using Chapter05.Business;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Chapter05.Repo {
    public interface ISampleFileReader
    {
        Task<IEnumerable<Book>> GetBooksAsync();
    }
}