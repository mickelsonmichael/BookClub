
using System.Collections.Generic;
using System.Threading.Tasks;
using Sample.Business;

namespace Sample.Repo
{
    public interface IPlaceholderRepo
    {
        Task<IEnumerable<Comment>> GetCommentsAsync();
    }
}