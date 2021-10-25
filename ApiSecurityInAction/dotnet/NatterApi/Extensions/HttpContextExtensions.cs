using Microsoft.AspNetCore.Http;

namespace NatterApi.Extensions
{
    public static class HttpContextExtensions
    {
        public static string? GetNatterUsername(this HttpContext context)
        {
            return (string?)context.Items["NatterUsername"];
        }

        public static void SetNatterUsername(this HttpContext context, string username)
        {
            context.Items["NatterUsername"] = username;
        }
    }
}
