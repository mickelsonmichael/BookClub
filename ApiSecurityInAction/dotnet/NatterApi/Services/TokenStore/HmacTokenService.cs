using System;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public class HmacTokenService : ITokenService
    {
        public HmacTokenService(NatterDbContext context, ILogger<DatabaseTokenService> logger)
        {
            _delegate = new DatabaseTokenService(context, logger);
        }

        public string CreateToken(HttpContext context, Token token)
        {
            string tokenId = _delegate.CreateToken(context, token);
            
            byte[] tag = Hash(tokenId);

            return $"{tokenId}.{Convert.ToBase64String(tag)}";
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            (bool valid, string realTokenId) = GetTokenId(tokenId);

            if (!valid)
            {
                return null;
            }

            return _delegate.ReadToken(context, realTokenId);
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            (bool valid, string realTokenId) = GetTokenId(tokenId);

            if (!valid)
            {
                return;
            }

            _delegate.DeleteToken(context, realTokenId);
        }

        private (bool valid, string tokenId) GetTokenId(string tokenId)
        {
            int splitIndex = tokenId.LastIndexOf('.');

            if (splitIndex < 0)
            {
                return (false, string.Empty);
            }

            string realTokenId = tokenId.Substring(0, splitIndex);

            byte[] providedHmac = Convert.FromBase64String(tokenId.Substring(splitIndex + 1));
            byte[] computedHmac = Hash(realTokenId);

            if (CryptographicOperations.FixedTimeEquals(providedHmac, computedHmac))
            {
                return (false, string.Empty);
            }

            return (true, realTokenId);
        }

        public Task ClearExpiredTokens()
        {
            return _delegate.ClearExpiredTokens();
        }

        private byte[] Hash(string value)
        {
            byte[] unhashedBytes = Encoding.UTF8.GetBytes(value);

            return _hmac.ComputeHash(unhashedBytes);
        }

        private readonly ITokenService _delegate;
        private readonly HMACSHA256 _hmac = new(); // randomly generate the key
    }
}
