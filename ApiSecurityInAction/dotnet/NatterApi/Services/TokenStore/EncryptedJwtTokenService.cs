using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using JWT;
using JWT.Algorithms;
using JWT.Builder;
using JWT.Serializers;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public class EncryptedJwtTokenService : ISecureTokenService
    {
        public Task ClearExpiredTokens()
        {
            // TODO
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            JwtBuilder jwtBuilder = JwtBuilder.Create()
                .Subject(token.Username)
                .Audience("https://localhost:4567")
                .ExpirationTime(token.Expiration)
                .WithAlgorithm(new HMACSHA256Algorithm())
                .WithSerializer(new JsonNetSerializer())
                .WithUrlEncoder(new JwtBase64UrlEncoder())
                .WithSecret("my_secret");

            foreach ((string key, string value) in token.Attributes)
            {
                jwtBuilder.AddClaim(key, value);
            }

            return jwtBuilder.Encode();
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            throw new System.NotImplementedException();
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            IJsonSerializer jsonSerializer = new JsonNetSerializer();

            IJwtDecoder decoder = new JwtDecoder(
                jsonSerializer,
                new JwtValidator(jsonSerializer, new UtcDateTimeProvider()),
                new JwtBase64UrlEncoder(),
                new HMACSHA256Algorithm()
            );

            var jwt = decoder.DecodeToObject<IDictionary<string, object>>(
                tokenId,
                key: "my_secret",
                verify: true
            );

            Token token = new(
                new DateTime((long)(double)jwt["exp"]).AddTicks(DateTime.UnixEpoch.Ticks),
                (string)jwt["sub"]
            );

            string[] ignore = new[] { "sub", "exp", "aud" };

            foreach (KeyValuePair<string, object> value in jwt.Where(val => !ignore.Contains(val.Key)))
            {
                token.Attributes.Add((value.Key, (string)value.Value));
            }

            return token;
        }
    }
}