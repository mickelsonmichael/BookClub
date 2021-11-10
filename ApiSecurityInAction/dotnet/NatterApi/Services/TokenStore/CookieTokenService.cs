using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
using NatterApi.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

        public string CreateToken(HttpRequest request)
        {
            //avoid session fixation
            var user = request.HttpContext.GetNatterUsername();
            if (user != null)
                DeleteToken(request);

            var userName = request.HttpContext.Items["NatterUsername"]?.ToString();
            var expiry = DateTime.Now.AddMinutes(10);
            var sessionCookie = request.HttpContext.SetNatterSession(userName, expiry);

            var session = new Session();
            session.SessionCookieId = sessionCookie.Id;
            session.CreatedDate = DateTime.Now;
            session.ExpireDate = expiry;
            session.UserName = userName;
            _dbContext.Add(session);
            _dbContext.SaveChanges();

            return Convert.ToBase64String(Encoding.ASCII.GetBytes(sessionCookie.Id)); //double submit

        }

        public ISession? ReadToken(HttpContext context, string tokenId)
        {
            //double-submit check
            var savedTokenId = _dbContext.Sessions.Where(x => x.UserName == context.GetNatterUsername()).FirstOrDefault();
            if (Convert.FromBase64String(tokenId) == Encoding.ASCII.GetBytes(savedTokenId.SessionCookieId))
                return context.Session;

            return null;
        }

        public void DeleteToken(HttpRequest request)
        {
            request.HttpContext.Session.Clear();
        }


        private readonly NatterDbContext _dbContext;
        private readonly ILogger<AuthService> _logger;
    }
}
