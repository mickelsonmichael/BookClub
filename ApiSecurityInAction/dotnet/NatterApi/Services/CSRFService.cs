using System;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http;

namespace NatterApi.Services
{
    /// <summary>
    /// <para>
    /// ASP.NET has a fundamental flaw in that you cannot modify the session ID
    /// once it's created. It is therefore particularly vulnerable to session
    /// fixation attacks. The OWASP foundation recommends the following strategy
    /// be injected after a user has been authenticated to ensure they are valid.
    /// </para>
    /// <para>https://stackoverflow.com/questions/47335370/session-fixation-change-sessionid-on-asp-net-core-2</para>
    /// <para>https://owasp.org/www-community/controls/Session_Fixation_Protection</para>
    /// <para>
    /// This service creates a cookie and equivalent session value. If the user doesn't
    /// send in the a cookie value that is equal to the current session value, then it
    /// should return `false` and the user should be force to re-log in.
    /// </para>
    /// <para>
    /// In the book, however, the CSRFToken performs a very similar job, so we can just
    /// utilize that instead. The only difference is that no value is stored in the session
    /// but instead the value compared is the <seealso ref="ISession.Id" />.
    /// </para>
    /// </summary>
    public static class CSRFService
    {
        private const string CookieName = "__Host_CSRFToken";

        public static string CreateToken(HttpContext context)
        {
            return GetHashedValue(context.Session.Id);
        }

        public static void SetToken(HttpContext context, string token)
        {
            CookieOptions cookieOptions = new()
            {
                Path = "/",
                HttpOnly = true, // value cannot be read from JS
                Secure = true, // the cookie can only be sent over HTTPS\
                SameSite = SameSiteMode.Strict // only send on same-site requests
            };

            context.Response.Cookies.Append(CookieName, token, cookieOptions);
        }

        public static string? GetToken(HttpContext context)
        {
            return context.Request.Cookies[CookieName];
        }

        public static bool VerifyToken(HttpContext context)
        {
            string? cookieValue = context.Request.Cookies[CookieName];

            if (string.IsNullOrWhiteSpace(cookieValue))
            {
                return false;
            }

            byte[] cookieBytes = Convert.FromBase64String(cookieValue);
            byte[] sessionBytes = Convert.FromBase64String(context.Session.Id);

            return CryptographicOperations.FixedTimeEquals(cookieBytes, sessionBytes);
        }

        private static string GetHashedValue(string value)
        {
            byte[] idBytes = Encoding.UTF8.GetBytes(value);

            using SHA256 sha256 = SHA256.Create();

            byte[] hash = sha256.ComputeHash(idBytes);

            return Convert.ToBase64String(hash);
        }
    }
}
