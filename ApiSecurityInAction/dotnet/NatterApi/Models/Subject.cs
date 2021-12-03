using System.Collections.Generic;

namespace NatterApi.Models
{
    /// <summary>
    /// 8.3.2 Implementing ABAC decisions
    /// </summary>
    public class Subject : Dictionary<string, object?>
    {
        public Subject(IDictionary<string, object?> subs)
            : base(subs)
        {
        }
    }
}
