﻿using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
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

        public string CreateToken(HttpContext context, Token token)
        {
            _logger.LogDebug("Updating session from token.");

            context.Session.SetString("username", token.Username);
            context.Session.SetString("expiry", token.Expiration.ToString("G"));
            context.Session.SetString("attrs", JsonSerializer.Serialize(token.Attributes));

            return context.Session.Id;
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            _logger.LogDebug("Reading token <{TokenId}> from session.", tokenId);

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

            return token;
        }

        public void DeleteToken(HttpContext context)
        {
            _logger.LogDebug("Deleting token.");
            
            context.Session.Clear();
        }

        private readonly ILogger<AuthService> _logger;
    }
}
