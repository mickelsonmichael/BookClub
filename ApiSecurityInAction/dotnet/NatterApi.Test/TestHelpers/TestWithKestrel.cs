using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Hosting;
using NatterApi.Models.Requests;
using Newtonsoft.Json;

namespace NatterApi.Test.TestHelpers
{
    public class TestWithKestrel
    {
        public TestWithKestrel()
        {
            _host = new HostBuilder().ConfigureWebHost(webBuilder =>
            {
                webBuilder
                    .UseTestServer()
                    .UseStartup<Startup>();
            }).Start();
        }

        protected async Task<HttpResponseMessage> GetResponse(HttpRequestMessage request = null, bool ensureStatusCode = true)
        {
            HttpClient client = _host.GetTestClient();

            HttpResponseMessage response = await client.SendAsync(request ?? GetDefault());

            if (ensureStatusCode)
            {
                response.EnsureSuccessStatusCode();
            }

            return response;
        }

        private static HttpRequestMessage GetDefault()
        {
            HttpRequestMessage requestMessage = new(HttpMethod.Post, "/spaces");

            CreateSpaceRequest createRequest = new("space-1", "test-apparatus");

            string json = JsonConvert.SerializeObject(createRequest);

            requestMessage.Content = new StringContent(json, Encoding.UTF8, "application/json");

            return requestMessage;
        }

        protected readonly IHost _host;
    }
}
