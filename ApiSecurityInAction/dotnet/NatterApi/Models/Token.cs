using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NatterApi.Models
{
    public record Token
    {
        public string Subject { get; set; }
        public DateTime Expiry { get; set; }
        public List<TokenOptions> Attributes { get; set; }

        public Token()
        {
            Attributes = new List<TokenOptions>();
        }
    }

    public class TokenOptions
    {
        public string Attribute { get; set; }
    }
}
