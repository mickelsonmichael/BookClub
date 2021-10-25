using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Test.TestHelpers;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter02Tests
    {
        private readonly WebApplicationFactory<Startup> _factory = new();

        [Fact]
        public async Task XSSHeaderDisabled()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            HttpAssert.HasHeader("X-XSS-Protection", response, expectedValue: "0");
        }

        [Fact]
        public async Task ContentTypeIncludesCharset()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            string contentType = response.Content.Headers.GetValues("Content-Type").First();

            Assert.Matches(@"charset=[uU][tT][fF]-8", contentType);
        }

        [Fact]
        public async Task NoSniff()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            HttpAssert.HasHeader("X-Content-Type-Options", response, expectedValue: "nosniff");
        }

        [Fact]
        public async Task PreventFrames()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            HttpAssert.HasHeader("X-Frame-Options", response, expectedValue: "DENY");

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Matches(@"frame-ancestors\s*'none'", securityPolicy);
        }

        [Fact]
        public async Task NoStoreCacheControl()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            HttpAssert.HasHeader("Cache-Control", response, expectedValue: "no-store");
        }

        [Fact]
        public async Task DisablesDefaultSrc()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Matches(@"default-src\s*'none'", securityPolicy);
        }

        [Fact]
        public async Task Sandbox()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Contains("sandbox", securityPolicy);
        }

        [Fact]
        public async Task DoesNotContainServerInfo()
        {
            HttpClient client = _factory.CreateClient();

            HttpResponseMessage response = await RequestHelpers.GetResponse(client).ConfigureAwait(false);

            HttpAssert.HasHeader("Server", response, expectedValue: string.Empty);
        }
    }
}