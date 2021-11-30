using System;
using System.Collections.Generic;
using System.Linq;

namespace NatterApi.Models.Token
{
    public record Token(
        DateTime Expiration,
        string Username
    )
    {
        public string Id { get; set; } = string.Empty;
        public List<(string, string)> Attributes = new();

        public void AddAttribute(string key, string? value)
        {
            Attributes.Add((key, value ?? string.Empty));
        }

        public static Token FromClaims(IDictionary<string, object> claims)
        {
            DateTime exp = DateTime.UnixEpoch.AddSeconds((long)claims["exp"]);
            string username = (string)claims["username"];

            Token token = new(exp, username);

            string[] ignore = new[] { "exp", "username", "iss", "aud" };
            foreach (var claim in claims.Where(c => !ignore.Contains(c.Key)))
            {
                token.AddAttribute(claim.Key, claim.Value.ToString());
            }

            return token;
        }
    }
}
