using System;
using System.Collections.Generic;

namespace NatterApi.Models.Token
{
    public record Token(
        DateTime Expiration,
        string Username
    )
    {
        public string Id { get; set; } = string.Empty;
        public List<(string, string)> Attributes = new();
    }
}
