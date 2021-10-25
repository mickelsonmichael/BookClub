
using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Extensions;
using NatterApi.Models;
using NatterApi.Services;

namespace NatterApi.Middleware
{
    public class AuthMiddleware
    {
        public AuthMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context, AuthService authService)
        {
            IEnumerable<string> authHeaders = context.Request.Headers["Authorization"];

            if (authHeaders == null || !authHeaders.Any() || !authHeaders.First().StartsWith("Basic"))
            {
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

            if (authService.TryLogin(username, password))
            {
                context.SetNatterUsername(username);
            }

            await _next(context);
        }

        private readonly RequestDelegate _next;
    }
}