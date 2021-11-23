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
        public EncryptedJwtTokenService(DatabaseTokenService allowList)
        {
            _allowList = allowList;
        }

        public Task ClearExpiredTokens()
        {
            // TODO
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            Token allowListToken = new(token.Expiration, token.Username);

            string jwtId = _allowList.CreateToken(context, allowListToken);

            JwtBuilder jwtBuilder = JwtBuilder.Create()
                .Id(jwtId)
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
            IDictionary<string, object> jwt = DecodeJwt(tokenId);

            string id = (string)jwt["jti"];

            _allowList.DeleteToken(context, id);
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            IDictionary<string, object> jwt = DecodeJwt(tokenId);

            string id = (string)jwt["jti"];

            if (_allowList.ReadToken(context, id) == null)
            {
                return null;
            }

            long ticks = (long)(double)jwt["exp"];
            DateTime exp = DateTime.UnixEpoch.AddSeconds(ticks);

            Token token = new(
                exp,
                (string)jwt["sub"]
            );

            string[] ignore = new[] { "sub", "exp", "aud" };

            foreach (KeyValuePair<string, object> value in jwt.Where(val => !ignore.Contains(val.Key)))
            {
                token.Attributes.Add((value.Key, (string)value.Value));
            }

            return token;
        }

        private IDictionary<string, object> DecodeJwt(string token)
        {
            IJsonSerializer jsonSerializer = new JsonNetSerializer();

            IJwtDecoder decoder = new JwtDecoder(
                jsonSerializer,
                new JwtValidator(jsonSerializer, new UtcDateTimeProvider()),
                new JwtBase64UrlEncoder(),
                new HMACSHA256Algorithm()
            );

            return decoder.DecodeToObject<IDictionary<string, object>>(
                token,
                key: "my_secret",
                verify: false
            );
        }

        private readonly DatabaseTokenService _allowList;
    }
}