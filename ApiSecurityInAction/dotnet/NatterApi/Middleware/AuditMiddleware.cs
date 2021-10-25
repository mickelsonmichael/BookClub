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
        public AuditMiddleware(RequestDelegate next, IServiceProvider serviceProvider)
        {
            _next = next;
            _services = serviceProvider;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            await LogRequestAsync(context).ConfigureAwait(false);

            await _next(context).ConfigureAwait(false);

            await LogResponseAsync(context).ConfigureAwait(false);
        }

        private async Task LogRequestAsync(HttpContext context)
        {
            HttpRequest request = context.Request;
            string? username = context.GetNatterUsername();

            AuditMessage auditMessage = AuditMessage.AuditRequest(
                request.Method,
                request.Path,
                username
            );

            using IServiceScope scope = _services.CreateScope();
            using NatterDbContext dbContext = scope.ServiceProvider.GetRequiredService<NatterDbContext>();

            dbContext.AuditLog.Add(auditMessage);
            await dbContext.SaveChangesAsync().ConfigureAwait(false);
        }

        private async Task LogResponseAsync(HttpContext context)
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

            using IServiceScope scope = _services.CreateScope();
            using NatterDbContext dbContext = scope.ServiceProvider.GetRequiredService<NatterDbContext>();

            dbContext.AuditLog.Add(auditMessage);
            await dbContext.SaveChangesAsync().ConfigureAwait(false);
        }

        private readonly RequestDelegate _next;
        private readonly IServiceProvider _services;
    }
}