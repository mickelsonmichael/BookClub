using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public interface ITokenService
    {
        string CreateToken(HttpContext context, Token token);
        Token? ReadToken(HttpContext context, string tokenId);
        void DeleteToken(HttpContext context, string tokenId);
    }
}
