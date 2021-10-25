using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
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
        public AuditMiddleware(RequestDelegate next, NatterDbContext dbContext)
        {
            _next = next;
            _dbContext = dbContext;
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

            _dbContext.AuditLog.Add(auditMessage);
            await _dbContext.SaveChangesAsync().ConfigureAwait(false);
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

            _dbContext.AuditLog.Add(auditMessage);
            await _dbContext.SaveChangesAsync().ConfigureAwait(false);
        }

        private readonly RequestDelegate _next;
        private readonly NatterDbContext _dbContext;
    }
}