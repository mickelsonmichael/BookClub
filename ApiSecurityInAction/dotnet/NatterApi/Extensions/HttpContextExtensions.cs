using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using System;
using System.Text.Json;

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

        public static int? GetSpaceId(this HttpContext context)
        {
            string[] parts = context.Request.Path.ToString().Trim('/').Split("/");

            if (parts[0] == "spaces" && parts.Length > 1 && int.TryParse(parts[1], out int spaceId))
            {
                return spaceId;
            }

            return null;
        }
    }
}
