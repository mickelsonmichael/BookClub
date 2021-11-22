using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace NatterApi.Test.TestHelpers
{
    public class SessionHelper
    {
        public string Token { get; set; }

        public SessionHelper(HttpClient client)
        {
            _httpClient = client;
        }

        public SessionHelper Register(string username, string password)
        {
            var registerRequest = new
            {
                username,
                password
            };

            HttpRequestMessage requestMessage = new(HttpMethod.Post, "/users")
            {
                Content = ToJsonContent(registerRequest)
            };

            HttpResponseMessage response = _httpClient.SendAsync(requestMessage).GetAwaiter().GetResult();

            response.EnsureSuccessStatusCode();

            _username = username;
            _password = password;

            return this;
        }

        public SessionHelper Login()
        {
            Debug.Assert(!string.IsNullOrWhiteSpace(_username));
            Debug.Assert(!string.IsNullOrWhiteSpace(_password));

            HttpRequestMessage request = new(HttpMethod.Post, "/sessions")
            {
                Content = new StringContent("")
            };

            request.Headers.Authorization = GetBasicAuthHeader();

            HttpResponseMessage response = _httpClient.SendAsync(request).GetAwaiter().GetResult();

            response.EnsureSuccessStatusCode();

            Token = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();

            return this;
        }

        public async Task<(HttpResponseMessage responseMessage, string spaceId)> CreateSpace(string name, string owner)
        {
            Debug.Assert(!string.IsNullOrWhiteSpace(Token));

            var createSpaceObj = new
            {
                name,
                owner
            };

            HttpRequestMessage request = new(HttpMethod.Post, "/spaces")
            {
                Content = ToJsonContent(createSpaceObj)
            };

            request.Headers.Authorization = GetBearerAuthHeader();

            HttpResponseMessage response = await _httpClient.SendAsync(request);

            string spaceId = response.Headers.Location?.OriginalString.Split('/').Last() ?? string.Empty;

            return (response, spaceId);
        }

        public SessionHelper GetSpace(string spaceId)
        {
            HttpRequestMessage request = new(HttpMethod.Get, $"/spaces/{spaceId}");

            HttpResponseMessage response = _httpClient.SendAsync(request).GetAwaiter().GetResult();

            response.EnsureSuccessStatusCode();

            return this;
        }

        private AuthenticationHeaderValue GetBasicAuthHeader()
        {
            Debug.Assert(!string.IsNullOrWhiteSpace(_username));
            Debug.Assert(!string.IsNullOrWhiteSpace(_password));

            string joined = $"{_username}:{_password}";

            string encoded = Convert.ToBase64String(
                Encoding.UTF8.GetBytes(joined)
            );

            return new AuthenticationHeaderValue("Basic", encoded);
        }

        private AuthenticationHeaderValue GetBearerAuthHeader()
        {
            Debug.Assert(!string.IsNullOrWhiteSpace(Token));

            return new AuthenticationHeaderValue("Bearer", Token);
        }
        
        private HttpContent ToJsonContent<T>(T obj) => new StringContent(ToJson(obj), Encoding.UTF8, "application/json");
        private string ToJson<T>(T obj) => JsonSerializer.Serialize(obj);
        private T FromJson<T>(string json) => JsonSerializer.Deserialize<T>(json);

        private string _username;
        private string _password;
        private readonly HttpClient _httpClient;
    }
}