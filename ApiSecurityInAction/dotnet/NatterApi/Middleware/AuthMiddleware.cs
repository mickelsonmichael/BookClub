
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Extensions;
using NatterApi.Services;

namespace NatterApi.Middleware
{
    public class AuthMiddleware
    {
        public AuthMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context, AuthService authService, ILogger<AuthMiddleware> logger)
        {
            IEnumerable<string> authHeaders = context.Request.Headers["Authorization"];

            if (authHeaders?.Any() != true || !authHeaders.First().StartsWith("Basic"))
            {
                logger.LogInformation("Skipping authorization.");

                // skip auth
                await _next(context);

                return;
            }

            if (authHeaders.Count() != 1)
            {
                throw new ArgumentException("Invalid Authorization header.");
            }

            string base64String = authHeaders.First().Substring("Basic ".Length);

            string credentials = Encoding.UTF8.GetString(
                Convert.FromBase64String(base64String)
            );

            string[] components = credentials.Split(":");

            if (components.Length != 2)
            {
                throw new ArgumentException("Invalid Authorization header.");
            }

            string username = components[0];
            string password = components[1];

            logger.LogDebug("Attempting to login {Username}.", username);

            if (!authService.TryLogin(username, password))
            {
                logger.LogDebug("Login attempt unsuccessful.");

                await _next(context);

                return;
            }

            context.SetNatterUsername(username);

            if (SessionFixationService.VerifyFixationToken(context))
            {
                logger.LogWarning("Old session detected. Invalidating.");

                context.Session.Clear();
            }

            await _next(context);
        }

        private readonly RequestDelegate _next;
    }
}
