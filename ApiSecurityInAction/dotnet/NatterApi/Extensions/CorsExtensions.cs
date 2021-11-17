using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;

namespace NatterApi.Extensions
{
    public static class CorsExtensions
    {
        public const string CorsPolicyName = "NatterApiCorsPolicy";
        public static IServiceCollection AddNatterCors(this IServiceCollection services)
        {
            return services.AddCors();
        }

        public static IApplicationBuilder UseNatterCors(this IApplicationBuilder app, params string[] allowedOrigins)
        {
            return app.UseCors(policy =>
                policy.WithOrigins(allowedOrigins)
                    .AllowAnyHeader()
                    .AllowAnyMethod()
                    .AllowCredentials()
            );
        }
    }
}
