using System.Collections.Generic;
using AspNetCoreRateLimit;
using Microsoft.Extensions.DependencyInjection;

namespace NatterApi.Extensions
{
    public static class IServiceCollectionExtensions
    {
        /// <summary>
        /// Section 3.2
        /// This section adds the AspNetCoreRateLimit library
        /// to the middleware pipeline.
        /// </summary>
        public static IServiceCollection AddRateLimiting(this IServiceCollection services)
        {
            return services.AddOptions()
                .AddMemoryCache()
                .Configure<IpRateLimitOptions>((options) =>
                {
                    options.GeneralRules = new List<RateLimitRule>
                        {
                            new()
                            {
                                Endpoint = "*",
                                Period = "1s",
                                Limit = 2
                            }
                        };
                })
                .Configure<IpRateLimitPolicies>(policies =>
                {
                    // configure ip policies
                })
                .AddInMemoryRateLimiting()
                .AddSingleton<IRateLimitConfiguration, RateLimitConfiguration>();
        }
    }
}
