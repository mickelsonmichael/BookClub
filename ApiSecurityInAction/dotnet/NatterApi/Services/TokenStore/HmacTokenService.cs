using System;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;

namespace NatterApi.Services.TokenStore
{
    public sealed class HmacTokenService : ISecureTokenService
    {
        private HmacTokenService(ITokenService tokenService)
        {
            _delegate = tokenService;
        }

        public static ISecureTokenService Create(IConfidentialTokenService internalService)
            => new HmacTokenService(internalService);

        public static IAuthenticatedTokenService Create(ITokenService internalService)
            => new HmacTokenService(internalService);

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

            string realTokenId = tokenId[..splitIndex];

            byte[] providedHmac = Convert.FromBase64String(tokenId[(splitIndex + 1)..]);
            byte[] computedHmac = Hash(realTokenId);

            return CryptographicOperations.FixedTimeEquals(providedHmac, computedHmac)
                ? (true, realTokenId)
                : (false, string.Empty);
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
        private readonly HMACSHA256 _hmac = new(Encoding.UTF8.GetBytes("SomeSecureKey"));
    }
}
