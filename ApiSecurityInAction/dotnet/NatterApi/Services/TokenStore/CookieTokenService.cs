using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
using NatterApi.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NatterApi.Services.TokenStore
{
    public class CookieTokenService : ITokenService
    {
        public CookieTokenService(NatterDbContext dbContext, ILogger<AuthService> logger)
        {
            _dbContext = dbContext;
            _logger = logger;
        }

        public string CreateToken(HttpRequest request, Token token)
        {
            // avoid session fixation
            // https://stackoverflow.com/questions/2402312/session-fixation-in-asp-net

            // if (request.HttpContext.Session.SessionID != null)
            // {
            //     AbandonSession(request);
            // }

            request.HttpContext.SetNatterSession(token);

            //double submit

            return request.HttpContext.Session.Id;
        }

        private void AbandonSession(HttpRequest request)
        {
            ISession session = request.HttpContext.Session;

            session.Clear();

            // if (request.Cookies["NatterCookie"] != null)
            // {
            //     HttpResponse response = request.HttpContext.Response;

            //     response.Cookies["NatterCookie"].Value = string.Empty;
            //     response.Cookies["NatterCookie"].Expires = DateTime.Now.AddMonths(-20);
            // }
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            ISession session = context.Session;

            if (session == null || !session.Keys.Any())
            {
                return null;
            }

            string username = session.GetString("username");
            DateTime expiry = DateTime.Parse(session.GetString("expiry"));
            string attributesJson = session.GetString("attrs");

            Token token = new(expiry, username);

            token.Attributes.AddRange(
                JsonSerializer.Deserialize<(string, string)[]>(attributesJson)
            );


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


        private readonly NatterDbContext _dbContext;
        private readonly ILogger<AuthService> _logger;
    }
}
