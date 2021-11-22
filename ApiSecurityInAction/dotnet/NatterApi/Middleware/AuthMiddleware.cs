
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Extensions;
using NatterApi.Models.Token;
using NatterApi.Services;
using NatterApi.Services.TokenStore;

namespace NatterApi.Middleware
{
    public class AuthMiddleware
    {
        public AuthMiddleware(RequestDelegate next, ILogger<AuthMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context, AuthService authService, ITokenService tokenService)
        {
            if (!HasAuthenticationHeader(context))
            {
                _logger.LogInformation("Skipping authorization.");

                // skip auth
                await _next(context);

                return;
            }

            (string authType, string headerValue) = GetAuthenticationMethod(context);

            (bool isSuccessful, string username) = authType switch
            {
                "Basic" => TryBasicAuthentication(headerValue, authService),
                "Bearer" => TryBearerAuthentication(headerValue, tokenService, context),

                _ => throw new InvalidOperationException("Unexpected authentication type provided.")
            };

            if (isSuccessful)
            {
                context.SetNatterUsername(username);

                await _next(context);
            }
            else
            {
                context.Response.StatusCode = (int)HttpStatusCode.Unauthorized;
            }
        }

        private (bool isSuccess, string username) TryBasicAuthentication(string headerValue, AuthService authService)
        {
            _logger.LogDebug("Using 'Basic' authentication scheme.");

            string base64String = headerValue.Substring("Basic ".Length);

            string credentials = Encoding.UTF8.GetString(
                Convert.FromBase64String(base64String)
            );

            string[] components = credentials.Split(":");

            if (components.Length != 2)
            {
                throw new ArgumentException("Invalid Basic authorization.");
            }

            string username = components[0];
            string password = components[1];

            _logger.LogDebug("Attempting to login {Username}.", username);

            return (authService.TryLogin(username, password), username);
        }

        private (bool isSuccess, string username) TryBearerAuthentication(string headerValue, ITokenService tokenService, HttpContext context)
        {
            string tokenId = headerValue.Substring("Bearer ".Length);

            Token? token = tokenService.ReadToken(context, tokenId);

            if (token == null || token.Expiration < DateTime.Now)
            {
                context.Response.Headers["WWW-Authenticate"] = @"Bearer error=""invalid_token"" error_description=""expired""";

                return (isSuccess: false, username: string.Empty);
            }
            else
            {
                foreach ((string key, string value) in token.Attributes)
                {
                    context.Items[key] = value;
                }

                return (isSuccess: true, token.Username);
            }
        }

        private bool HasAuthenticationHeader(HttpContext context)
            => context.Request.Headers["Authentication"].Count == 1;

        private (string type, string value) GetAuthenticationMethod(HttpContext context)
        {
            string value = context.Request.Headers["Authentication"].First();

            if (value.StartsWith("Basic"))
            {
                return ("Basic", value);
            }
            else if (value.StartsWith("Bearer"))
            {
                return ("Bearer", value);
            }
            else
            {
                return (string.Empty, value);
            }
        }

        private readonly RequestDelegate _next;
        private readonly ILogger<AuthMiddleware> _logger;
    }
}
