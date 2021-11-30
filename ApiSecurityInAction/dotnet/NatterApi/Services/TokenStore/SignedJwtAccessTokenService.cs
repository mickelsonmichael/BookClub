using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Web;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using NatterApi.Models.Token;
using NatterApi.Options;
using JWT;
using JWT.Algorithms;
using System.Security.Cryptography;
using System.Linq;
using JWT.Serializers;
using JWT.Builder;
using Org.BouncyCastle.Crypto.Generators;
using Org.BouncyCastle.Security;
using System.Numerics;
using Microsoft.IdentityModel.Tokens;
using JWT.Exceptions;

namespace NatterApi.Services.TokenStore
{
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
            // todo
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
