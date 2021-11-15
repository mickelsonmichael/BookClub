using System;
using System.Security.Cryptography;
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
    /// </summary>
    public static class SessionFixationService
    {
        private const string CookieName = "__HostNatterAntiFixation";

        public static void CreateFixationToken(HttpContext context)
        {
            string value = GetRandomValue();

            CookieOptions cookieOptions = new()
            {
                Path = "/",
                HttpOnly = true, // value cannot be read from JS
                Secure = true, // the cookie can only be sent over HTTPS\
                SameSite = SameSiteMode.Strict // only send on same-site requests
            };

            context.Response.Cookies.Append(CookieName, value, cookieOptions);
            context.Session.SetString(CookieName, value);

            static string GetRandomValue()
            {
                using RNGCryptoServiceProvider random = new();

                var data = new byte[32];

                random.GetBytes(data);

                return Convert.ToBase64String(data);
            }
        }

        public static bool VerifyFixationToken(HttpContext context)
        {
            string? cookieValue = context.Request.Cookies[CookieName];

            string sessionValue = context.Session.GetString(CookieName);

            return cookieValue == sessionValue;
        }
    }
}
