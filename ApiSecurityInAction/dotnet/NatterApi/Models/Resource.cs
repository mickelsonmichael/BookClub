using System.Collections.Generic;

namespace NatterApi.Models
{
    /// <summary>
    /// 8.3.2 Implementing ABAC decisions
    /// </summary>
    public class Resource : Dictionary<string, object?>
    {
        public Resource(IDictionary<string, object?> res)
            : base(res)
        {
        }
    }
}
