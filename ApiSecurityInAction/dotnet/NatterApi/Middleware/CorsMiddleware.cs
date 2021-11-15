using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;
using NatterApi.Configuration;

namespace NatterApi.Middleware
{
    public class CorsMiddleware
    {
        public CorsMiddleware(RequestDelegate next, IOptions<CorsConfig> corsOptions)
        {
            _next = next;
            _allowedOrigins = corsOptions.Value.AllowedOrigins;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            string? origin = context.Request.Headers["Origin"].FirstOrDefault();

            if (IsOriginAllowed(origin))
            {
                context.Response.Headers["Access-Control-Allow-Origin"] = origin;
                context.Response.Headers["Access-Control-Allow-Credentials"] = "true";
                context.Request.Headers["Vary"] = "Origin";
            }

            if (!IsPreflightRequest(context.Request))
            {
                await _next(context);

                return;
            }

            if (!IsOriginAllowed(origin))
            {
                context.Response.StatusCode = (int)HttpStatusCode.Unauthorized;
            }
            else
            {
                context.Response.Headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-CSRF-Token";
                context.Response.Headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE";

                context.Response.StatusCode = (int)HttpStatusCode.Created;
            }
        }

        private bool IsOriginAllowed(string? origin) => !string.IsNullOrWhiteSpace(origin) && _allowedOrigins.Contains(origin);

        private static bool IsPreflightRequest(HttpRequest request)
        {
            return request.Method == "OPTIONS" && request.Headers.ContainsKey("Access-Control-Request-Method");
        }

        private readonly IReadOnlySet<string> _allowedOrigins;
        private readonly RequestDelegate _next;
    }
}