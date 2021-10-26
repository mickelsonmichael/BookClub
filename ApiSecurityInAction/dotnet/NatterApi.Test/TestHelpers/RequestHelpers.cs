using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using NatterApi.Models.Requests;

namespace NatterApi.Test.TestHelpers
{
    public static class RequestHelpers
    {
        public static async Task<HttpResponseMessage> GetResponse(
            HttpClient client,
            HttpRequestMessage request = null,
            bool ensureStatusCode = true
        )
        {
            request ??= GetDefault();

            HttpResponseMessage response = await client.SendAsync(request).ConfigureAwait(false);

            if (ensureStatusCode)
            {
                response.EnsureSuccessStatusCode();
            }

            return response;
        }

        public static Task<HttpResponseMessage> GetAsync(
            HttpClient client,
            string url,
            string username = null,
            string password = null
        )
        {
            HttpRequestMessage request = new(HttpMethod.Get, url);

            if (username != null && password != null)
            {
                AttachAuthentication(request, username, password);
            }

            return client.SendAsync(request);
        }

        public static Task<HttpResponseMessage> PostAsync<T>(
            HttpClient client,
            string url,
            T body,
            string username = null,
            string password = null
        )
        {
            HttpRequestMessage request = new(HttpMethod.Post, url)
            {
                Content = GetJsonContent(body)
            };

            if (username != null && password != null)
            {
                AttachAuthentication(request, username, password);
            }

            return client.SendAsync(request);
        }

        public static void AttachAuthentication(HttpRequestMessage message, string username, string password)
        {
            string creds = Base64Credentials(username, password);

            AuthenticationHeaderValue value = new("Basic", creds);

            message.Headers.Authorization = value;
        }
        
        private static string Base64Credentials(string username, string password)
        {
            string joined = $"{username}:{password}";

            return Convert.ToBase64String(
                Encoding.UTF8.GetBytes(joined)
            );
        }

        public static async Task RegisterUser(HttpClient client, string username, string password)
        {
            HttpRequestMessage request = new (HttpMethod.Post, "/users")
            {
                Content = GetJsonContent(new RegisterUser(username, password))
            };

            await GetResponse(client, request, ensureStatusCode: true);
        }

        private static StringContent GetJsonContent<T>(T obj)
        {
            string json = JsonSerializer.Serialize(obj);

            return new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        }

        private static HttpRequestMessage GetDefault()
        {
            return new(HttpMethod.Get, "/status");
        }
    }
}
