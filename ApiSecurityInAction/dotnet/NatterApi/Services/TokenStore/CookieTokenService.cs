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
            if (request.HttpContext.Items["username"] != null)
                DeleteToken(request);

            var userName = request.HttpContext.Items["NatterUsername"]?.ToString();
            var expiry = DateTime.Now.AddMinutes(10);
            var sessionCookie = request.HttpContext.SetNatterSessionCookie(userName, expiry);

            var session = new Session();
            session.SessionCookieId = sessionCookie.Id;
            session.CreatedDate = DateTime.Now;
            session.ExpireDate = expiry;
            session.UserName = userName;
            _dbContext.Add(session);
            _dbContext.SaveChanges();

            return Convert.ToBase64String(Encoding.ASCII.GetBytes(sessionCookie.Id)); //double submit

        }

        public bool ReadToken(HttpRequest request, string tokenId)
        {
            //double-submit check
            var savedTokenId = _dbContext.Sessions.Where(x => x.UserName == request.HttpContext.Items["username"]).FirstOrDefault();
            if(Convert.FromBase64String(tokenId) == Encoding.ASCII.GetBytes(savedTokenId.SessionCookieId))
                return true;

            return false;
        }

        public void DeleteToken(HttpRequest request)
        {
            //can't delete a session cookie
            request.HttpContext.Session.SetString("expiry", DateTime.Now.AddDays(-1).ToString("G"));
        }


        private readonly NatterDbContext _dbContext;
        private readonly ILogger<AuthService> _logger;
    }
}
