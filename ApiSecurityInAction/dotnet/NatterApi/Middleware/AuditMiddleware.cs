using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;

namespace NatterApi.Middleware
{
    /// <summary>
    /// Section 3.5
    /// Audit Logging
    /// </summary>
    public class AuditMiddleware
    {
        public AuditMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context, NatterDbContext dbContext)
        {
            await LogRequestAsync(context, dbContext);

            await _next(context).ConfigureAwait(false);

            await LogResponseAsync(context, dbContext);
        }

        private async Task LogRequestAsync(HttpContext context, NatterDbContext dbContext)
        {
            HttpRequest request = context.Request;
            string? username = context.GetNatterUsername();

            AuditMessage auditMessage = AuditMessage.AuditRequest(
                request.Method,
                request.Path,
                username
            );

            dbContext.AuditLog.Add(auditMessage);

            await dbContext.SaveChangesAsync();
        }

        private async Task LogResponseAsync(HttpContext context, NatterDbContext dbContext)
        {
            HttpRequest request = context.Request;
            HttpResponse response = context.Response;
            string? username = context.GetNatterUsername();

            AuditMessage auditMessage = AuditMessage.AuditResponse(
                request.Method,
                request.Path,
                username,
                response.StatusCode
            );

            dbContext.AuditLog.Add(auditMessage);

            await dbContext.SaveChangesAsync().ConfigureAwait(false);
        }

        private readonly RequestDelegate _next;
    }
}
