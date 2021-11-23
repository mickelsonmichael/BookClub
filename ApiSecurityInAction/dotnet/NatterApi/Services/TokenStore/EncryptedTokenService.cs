using System;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using Sodium;

namespace NatterApi.Services.TokenStore
{
    public class EncryptedTokenService : ITokenService
    {
        public EncryptedTokenService()
        {
            _delegate = new JsonTokenStore();
        }

        public Task ClearExpiredTokens()
        {
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            string tokenId = _delegate.CreateToken(context, token);

            byte[] hash = SealedPublicKeyBox.Create(tokenId, _key.PublicKey);

            return Convert.ToBase64String(hash);
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            byte[] bytes = Convert.FromBase64String(tokenId);

            byte[] originalTokenBytes = SealedPublicKeyBox.Open(bytes, _key);

            string originalTokenId = Encoding.UTF8.GetString(originalTokenBytes);

            _delegate.DeleteToken(context, originalTokenId);
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            byte[] bytes = Convert.FromBase64String(tokenId);

            byte[] originalTokenBytes = SealedPublicKeyBox.Open(bytes, _key);

            string originalTokenId = Encoding.UTF8.GetString(originalTokenBytes);

            return _delegate.ReadToken(context, originalTokenId);
        }

        private static readonly KeyPair _key = PublicKeyBox.GenerateKeyPair();
        private readonly ITokenService _delegate;
    }
}