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

namespace NatterApi.Services.TokenStore
{
    public class OAuth2TokenService : ISecureTokenService
    {
        public OAuth2TokenService(
            IOptions<KeycloakOptions> options,
            HttpClient httpClient,
            ILogger<OAuth2TokenService> logger
        )
        {
            _introspectionEndpoint = options.Value.IntrospectionEndpoint;
            _revokeEndpoint = options.Value.RevokeEndpoint;
            _httpClient = httpClient;
            _logger = logger;

            var encoder = Encoding.UTF8.GetEncoder();

            string credentials = HttpUtility.UrlEncodeUnicode(options.Value.ClientId)
                + ":" + HttpUtility.UrlEncodeUnicode(options.Value.Secret);

            _authorization = $"Basic {Convert.ToBase64String(Encoding.UTF8.GetBytes(credentials))}";
        }

        public Task ClearExpiredTokens()
        {
            throw new System.NotImplementedException();
        }

        public string CreateToken(HttpContext context, Token token)
        {
            throw new System.NotImplementedException();
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            IEnumerable<KeyValuePair<string?, string?>> form = new KeyValuePair<string?, string?>[]
            {
                new ("token", tokenId),
                new ("token_type_hint", "access_token")
            };

            HttpRequestMessage request = new(HttpMethod.Post, _revokeEndpoint)
            {
                Content = new FormUrlEncodedContent(form),
            };

            request.Headers.Add("Authorization", _authorization);

            HttpResponseMessage response = _httpClient.SendAsync(request).GetAwaiter().GetResult();

            if (response.StatusCode != HttpStatusCode.OK)
            {
                string errorMessage = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                _logger.LogCritical("Error attempting to revoke torkens via OAuth2. {ErrorMessage}", errorMessage);
            }
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            // ensure the token is under 1KB in size
            if (!Regex.IsMatch(tokenId, "[\\x20-\\x7E]{1,1024}"))
            {
                return null;
            }

            IEnumerable<KeyValuePair<string?, string?>> form = new KeyValuePair<string?, string?>[]
            {
                new ("token", tokenId),
                new ("token_type_hint", "access_token")
            };

            HttpRequestMessage request = new(HttpMethod.Post, _introspectionEndpoint)
            {
                Content = new FormUrlEncodedContent(form),
            };

            request.Headers.Add("Authorization", _authorization);

            HttpResponseMessage response = _httpClient.SendAsync(request).GetAwaiter().GetResult();

            if (response.StatusCode != HttpStatusCode.OK)
            {
                string errorMessage = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

                _logger.LogWarning("Error attempting to authenticate via OAuth2. {ErrorMessage}", errorMessage);

                return null;
            }

            string jsonString = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            JsonNode json = JsonSerializer.Deserialize<JsonNode>(jsonString)!;

            return (bool?)json["active"] == true ? ProcessResponse(json) : null;
        }

        private Token ProcessResponse(JsonNode response)
        {
            DateTime expiration = GetExpirationDate((long)(response["exp"]));
            string subject = (string)(response["username"]);

            Token token = new(expiration, subject);

            token.AddAttribute("scope", (string?)response["scope"]);
            token.AddAttribute("client_id", (string?)response["client_id"]);

            return token;
        }

        private DateTime GetExpirationDate(long ticks) => DateTime.UnixEpoch.AddSeconds(ticks);

        private readonly string _revokeEndpoint;
        private readonly string _introspectionEndpoint;
        private readonly string _authorization;
        private readonly HttpClient _httpClient;
        private readonly ILogger<OAuth2TokenService> _logger;
    }
}
