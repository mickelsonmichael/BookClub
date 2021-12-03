using System.Collections.Generic;

namespace NatterApi.Models
{
    /// <summary>
    /// 8.3.2 Implementing ABAC decisions
    /// </summary>
    public class ApiEnvironment : Dictionary<string, object?>
    {
        public ApiEnvironment()
            : base(new Dictionary<string, object?>())
        {
        }

        public ApiEnvironment(IDictionary<string, object?> envs)
            : base(envs)
        {
        }
    }
}
