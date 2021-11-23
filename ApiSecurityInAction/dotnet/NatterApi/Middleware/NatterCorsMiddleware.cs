using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace NatterApi.Middleware
{
    /// <summary>
    ///
    /// </summary>
    public class NatterCorsMiddleware
    {
        public NatterCorsMiddleware(RequestDelegate next, ILogger<NatterCorsMiddleware> logger)
        {
            _next = next;
            _allowedOrigins = new[] { "https://localhost:9999" };
            _logger = logger;

            _logger.LogInformation("Creating CORS middleware with allowed origins: {Origins}", string.Join(", ", _allowedOrigins));
        }

        public async Task InvokeAsync(HttpContext context)
        {
            _logger.LogInformation("checking for cors request");

            context.Items["__CorsMiddlewareWithEndpointInvoked"] = true;
            
            string? origin = context.Request.Headers["Origin"].FirstOrDefault();

            if (IsOriginAllowed(origin))
            {
                _logger.LogDebug("Allowed origin <{Origin}> detected. Attaching access control headers.", origin);

                context.Response.Headers["Access-Control-Allow-Origin"] = origin;

                // now that we're using bearer tokens, we don't need credentials
                // context.Response.Headers["Access-Control-Allow-Credentials"] = "true";

                context.Request.Headers["Vary"] = "Origin";
            }

            if (!IsPreflightRequest(context.Request))
            {
                await _next(context);

                return;
            }

            if (!IsOriginAllowed(origin))
            {
                _logger.LogWarning("Unauthorized CORS preflight request attempted from origin <{Origin}>.", origin);
                context.Response.StatusCode = (int)HttpStatusCode.Unauthorized;
            }
            else
            {
                _logger.LogDebug("Successfull preflight request from origin <{Origin}>, attaching headers to response.", origin);
                context.Response.Headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization";
                context.Response.Headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE";

                context.Response.StatusCode = (int)HttpStatusCode.Created;
            }
        }

        private bool IsOriginAllowed(string? origin) => !string.IsNullOrWhiteSpace(origin) && _allowedOrigins.Contains(origin);

        private static bool IsPreflightRequest(HttpRequest request)
        {
            return request.Method == "OPTIONS" && request.Headers.ContainsKey("Access-Control-Request-Method");
        }

        private readonly IReadOnlyList<string> _allowedOrigins;
        private readonly ILogger<NatterCorsMiddleware> _logger;
        private readonly RequestDelegate _next;
    }
}