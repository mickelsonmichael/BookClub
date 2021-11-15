using System.Collections.Generic;
using System.Linq;

namespace NatterApi.Configuration
{
    public record CorsConfig
    {
        public IReadOnlySet<string> AllowedOrigins { get; }

        public CorsConfig(params string[] allowedOrigins)
        {
            AllowedOrigins = allowedOrigins.ToHashSet();
        }
    }
}
