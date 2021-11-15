using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
using NatterApi.Extensions;
using System;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace NatterApi.Services.TokenStore
{
    public class CookieTokenService : ITokenService
    {
        public CookieTokenService(ILogger<AuthService> logger)
        {
            _logger = logger;
        }

        public string CreateToken(HttpRequest request, Token token)
        {
            // avoid session fixation
            // https://stackoverflow.com/questions/2402312/session-fixation-in-asp-net

            request.HttpContext.SetNatterSession(token);

            //double submit

            return request.HttpContext.Session.Id;
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            ISession session = context.Session;

            if (session?.Keys.Any() != true)
            {
                return null;
            }

            string username = session.GetString("username");
            DateTime expiry = DateTime.Parse(session.GetString("expiry"));
            string attributesJson = session.GetString("attrs");

            Token token = new(expiry, username);

            (string key, string value)[]? attributes = JsonSerializer.Deserialize<(string, string)[]>(attributesJson);

            if (attributes?.Any() == true)
            {
                token.Attributes.AddRange(attributes);
            }

            //double-submit check
            // var savedTokenId = _dbContext.Sessions.Where(x => x.UserName == context.GetNatterUsername()).FirstOrDefault();
            // if (Convert.FromBase64String(tokenId) == Encoding.ASCII.GetBytes(savedTokenId.SessionCookieId))
            //     return context.Session;

            return token;
        }

        public void DeleteToken(HttpRequest request)
        {
            request.HttpContext.Session.Clear();
        }

        private readonly ILogger<AuthService> _logger;
    }
}
