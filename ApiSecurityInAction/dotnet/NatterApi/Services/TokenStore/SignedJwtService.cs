using System;
using System.Security.Cryptography;
using System.Threading.Tasks;
using JWT.Algorithms;
using JWT.Builder;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public class SignedJwtService : ITokenService
    {
        public Task ClearExpiredTokens()
        {
            // can't do
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            JwtBuilder builder = JwtBuilder.Create()
                .WithAlgorithm(new HMACSHA256Algorithm())
                .WithSecret("my_secret")
                .Audience("https://localhost:4567")
                .Subject(token.Username)
                .ExpirationTime(token.Expiration);

            foreach ((string key, string value) in token.Attributes)
            {
                builder.AddClaim(key, value);
            }

            return builder.Encode();
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // TODO
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            // TODO
            return null;
        }
    }
}