using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public class JsonTokenStore : ITokenService
    {
        public Task ClearExpiredTokens()
        {
            // can't do it
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            var jsonObj = new
            {
                sub = token.Username,
                exp = (long)token.Expiration.Subtract(DateTime.UnixEpoch).TotalMilliseconds,
                attrs = token.Attributes
            };

            string json = JsonSerializer.Serialize(jsonObj);

            byte[] bytes = Encoding.UTF8.GetBytes(json);

            return Convert.ToBase64String(bytes);
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // TODO
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            byte[] bytes = Convert.FromBase64String(tokenId);

            string json = Encoding.UTF8.GetString(bytes);

            try
            {
                var obj = JsonSerializer.Deserialize<(
                    string sub,
                    long exp,
                    List<(string, string)> attrs
                )>(json);

                Token token = new(new DateTime(obj.exp), obj.sub);

                token.Attributes.AddRange(obj.attrs);

                return token;
            }
            catch
            {
                return null;
            }
        }
    }
}