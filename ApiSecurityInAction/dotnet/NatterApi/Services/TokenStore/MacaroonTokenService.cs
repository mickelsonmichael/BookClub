using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using NatterApi.Models.Token;
using Macaroons;
using System.Text;
using Microsoft.Extensions.Logging;
using NatterApi.Verifiers;

namespace NatterApi.Services.TokenStore
{
    public class MacaroonTokenService : ISecureTokenService
    {
        public static HMACSHA256 HmacKey
        {
            get
            {
                if (_hmacKey == null)
                {
                    byte[] rngBytes = new byte[256];
                    RandomNumberGenerator.Create().GetBytes(rngBytes);

                    _hmacKey = new HMACSHA256(rngBytes);
                }

                return _hmacKey;
            }
        }

        public MacaroonTokenService(DatabaseTokenService del, ILogger<MacaroonTokenService> logger)
        {
            _delegate = del;
            _logger = logger;

            _logger.LogInformation("Utilizing the macaroon token service.");
        }

        public Task ClearExpiredTokens()
        {
            return Task.CompletedTask;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            _logger.LogInformation("Creating a new macaroon");

            string id = _delegate.CreateToken(context, token);

            Macaroon macaroon = new(
                location: "",
                _key,
                identifier: id
            );

            // 9.3.4 Third-party caveats
            // macaroon.AddThirdPartyCaveat("https://service.example.com", "secret", "caveatid");

            return macaroon.Serialize();
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // do nothing
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            _logger.LogInformation("Reading a macaroon");

            Macaroon macaroon = Macaroon.Deserialize(tokenId);

            Verifier verifier = new();

            verifier.SatisfyGeneral(StandardCaveatVerifiers.ExpiresVerifier);

            verifier.SatisfyExact($"method = {context.Request.Method}");

            // 9.3.3
            // verify the since time on requests
            // this is a custom verifier in Natter.Verifiers
            verifier.AddSinceVerifier(context);

            VerificationResult result = macaroon.Verify(verifier, _key);

            if (result.Success)
            {
                return _delegate.ReadToken(context, macaroon.Identifier.ToString());
            }
            else
            {
                return null;
            }
        }

        private string _key => Encoding.UTF8.GetString(HmacKey.Key);
        private static HMACSHA256? _hmacKey;
        private readonly ITokenService _delegate;
        private readonly ILogger<MacaroonTokenService> _logger;
    }
}
