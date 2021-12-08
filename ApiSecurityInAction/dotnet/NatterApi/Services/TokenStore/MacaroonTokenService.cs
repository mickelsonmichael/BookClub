using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using Macaroons;
using System.Text;

namespace NatterApi.Services.TokenStore
{
    public class MacaroonTokenService : ISecureTokenService
    {
        public MacaroonTokenService(DatabaseTokenService del)
        {
            _delegate = del;

            byte[] rngBytes = new byte[256];
            RandomNumberGenerator.Create().GetBytes(rngBytes);

            _hmacKey = new HMACSHA256(rngBytes);
    }

        public Task ClearExpiredTokens()
        {
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            string id = _delegate.CreateToken(context, token);

            Macaroon macaroon = new(
                location: "",
                Encoding.UTF8.GetString(_hmacKey.Key),
                identifier: id
            );

            return macaroon.Serialize();l
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // do nothing
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            throw new System.NotImplementedException();
        }
        private readonly HMACSHA256 _hmacKey;
        private readonly ITokenService _delegate;
    }
}
