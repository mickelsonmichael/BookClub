using System.Collections.Generic;

namespace NatterApi.Models
{
    /// <summary>
    /// 8.3.2 Implementing ABAC decisions
    /// </summary>
    public class ApiAction : Dictionary<string, object?>
    {
        public ApiAction()
            : base(new Dictionary<string, object?>())
        {
        }

        public ApiAction(IDictionary<string, object?> actions)
            : base(actions)
        {
        }
    }
}
