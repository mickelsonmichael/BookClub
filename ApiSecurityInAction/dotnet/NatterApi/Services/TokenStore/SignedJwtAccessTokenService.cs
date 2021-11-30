using JWT.Algorithms;
using JWT.Builder;
using JWT.Exceptions;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using NatterApi.Models.Token;
using NatterApi.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace NatterApi.Services.TokenStore
{
    /// <summary>
    ///  http://localhost:8080/auth/realms/api-security/.well-known/openid-configuration
    ///  http://localhost:8080/auth/realms/api-security/protocol/openid-connect/certs
    ///  http://localhost:8080/auth/realms/api-security
    /// </summary>
    public class SignedJwtAccessTokenService : ISecureTokenService
    {
        public SignedJwtAccessTokenService(
            IOptions<KeycloakOptions> keycloakOpts,
            HttpClient httpClient
        )
        {
            _jwkSetUri = keycloakOpts.Value.JwkSetEndpoint;
            _expectedAudience = keycloakOpts.Value.Audience;
            _expectedIssuer = keycloakOpts.Value.Issuer;
            _httpClient = httpClient;
        }

        public string CreateToken(HttpContext context, Token token)
        {
            return string.Empty;
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            RSA publicKey = GetPublicKey();

            IDictionary<string, object> claims;

            try
            {
                claims = JwtBuilder.Create()
                            .MustVerifySignature()
                            .WithAlgorithm(new RS256Algorithm(publicKey))
                            .Decode<IDictionary<string, object>>(tokenId);
            }
            catch (SignatureVerificationException)
            {
                return null;
            }

            if (BadIssuer() || BadAudience())
            {
                return null;
            }

            return Token.FromClaims(claims);

            bool BadIssuer()
                => !claims["iss"].Equals(_expectedIssuer);

            bool BadAudience()
                => !claims["aud"].Equals(_expectedAudience);
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            // todo
        }

        public Task ClearExpiredTokens()
        {
            return Task.CompletedTask;
        }

        private RSA GetPublicKey()
        {
            var jsonString = _httpClient.GetStringAsync(_jwkSetUri).GetAwaiter().GetResult();

            JsonNode json = JsonSerializer.Deserialize<JsonNode>(jsonString)!;

            JsonArray keys = json["keys"]?.AsArray() ?? throw new InvalidOperationException($"Expected a keys property on the response.\n${json}");

            JsonNode ecsKey = keys.First(node =>
            {
                string alg = node!["alg"]!.GetValue<string>();
                string use = node!["use"]!.GetValue<string>();

                return "RS256".Equals(alg, StringComparison.OrdinalIgnoreCase)
                    && "sig".Equals(use, StringComparison.OrdinalIgnoreCase);
            }) ?? throw new InvalidOperationException($"Expected a key with RS256 algorithm. Algs: ${string.Join(", ", keys.Select(node => (string?)(node?["alg"] ?? "?")))}");

            JsonWebKey jsonWebKey = new(ecsKey.ToJsonString());

            RSAParameters publicKey = new()
            {
                Exponent = Base64UrlEncoder.DecodeBytes(jsonWebKey.E),
                Modulus = Base64UrlEncoder.DecodeBytes(jsonWebKey.N)
            };

            var prov = new RSACryptoServiceProvider();
            prov.ImportParameters(publicKey);

            return RSA.Create(publicKey);
        }

        private readonly string _expectedIssuer;
        private readonly string _expectedAudience;
        private readonly string _jwkSetUri;
        private readonly HttpClient _httpClient;
    }
}
