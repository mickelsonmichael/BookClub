using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public interface ITokenService
    {
        string CreateToken(HttpContext context, Token token);
        Token? ReadToken(HttpContext context, string tokenId);
        void DeleteToken(HttpContext context, string tokenId);
        /// 5.2.3 - Allow for the deletion of expired tokens using an automated, recurring task
        Task ClearExpiredTokens();
    }
}
