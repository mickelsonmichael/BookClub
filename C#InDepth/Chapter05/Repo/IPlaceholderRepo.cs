using System.Collections.Generic;
using System.Threading.Tasks;
using Chapter05.Business;

namespace Chapter05.Repo
{
    public interface IPlaceholderRepo
    {
        Task<IEnumerable<Comment>> GetCommentsAsync();
    }
}