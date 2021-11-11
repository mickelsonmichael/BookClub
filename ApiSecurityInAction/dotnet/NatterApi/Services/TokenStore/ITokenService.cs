using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NatterApi.Services.TokenStore
{
    public interface ITokenService
    {
        string CreateToken(HttpRequest request, Token token);
        Token? ReadToken(HttpContext context, string tokenId);
        void DeleteToken(HttpRequest request);
    }
}
