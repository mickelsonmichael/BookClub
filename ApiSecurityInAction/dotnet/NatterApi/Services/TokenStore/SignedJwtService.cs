using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using JWT.Algorithms;
using JWT.Builder;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using Newtonsoft.Json.Linq;

namespace NatterApi.Services.TokenStore
{
    public class SignedJwtService : IAuthenticatedTokenService
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
                .AddClaim("attrs", token.Attributes)
                .ExpirationTime(token.Expiration);

            return builder.Encode();
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // TODO
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            var jwt = JwtBuilder.Create()
                .WithAlgorithm(new HMACSHA256Algorithm())
                .WithSecret("my_secret")
                .Decode<IDictionary<string, object>>(tokenId);

            DateTime exp = DateTime.UnixEpoch
                .AddMilliseconds(
                    (long)(double)jwt["exp"]
                );

            Token token = new(
                exp,
                (string)jwt["sub"]
            );

            foreach (JObject val in ((JArray)jwt["attrs"]).Children<JObject>())
            {
                foreach (JProperty prop in val.Properties())
                {
                    token.Attributes.Add((
                        prop.Name,
                        (string?)prop.Value ?? string.Empty
                    ));
                }
            }

            return token;
        }
    }
}