using System;

namespace NatterApi.Models
{
    public record AuditMessage(
        int Id,
        string Method,
        string Path,
        string? Username,
        int? StatusCode,
        DateTime AuditTime
    )
    {
        public static AuditMessage AuditRequest(string method, string path, string? username)
            => new(0, method, path, username, null, DateTime.Now);

        public static AuditMessage AuditResponse(string method, string path, string? username, int statusCode)
            => new(0, method, path, username, statusCode, DateTime.Now);
    }
}
