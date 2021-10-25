using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using NatterApi.Test.TestHelpers;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter02Tests : TestWithKestrel
    {
        [Fact]
        public async Task XSSHeaderDisabled()
        {
            HttpResponseMessage response = await GetResponse();

            HttpAssert.HasHeader("X-XSS-Protection", response, expectedValue: "0");
        }

        [Fact]
        public async Task ContentTypeIncludesCharset()
        {
            HttpResponseMessage response = await GetResponse();

            string contentType = response.Content.Headers.GetValues("Content-Type").First();

            Assert.Matches(@"charset=[uU][tT][fF]-8", contentType);
        }

        [Fact]
        public async Task NoSniff()
        {
            HttpResponseMessage response = await GetResponse();

            HttpAssert.HasHeader("X-Content-Type-Options", response, expectedValue: "nosniff");
        }

        [Fact]
        public async Task PreventFrames()
        {
            HttpResponseMessage response = await GetResponse();

            HttpAssert.HasHeader("X-Frame-Options", response, expectedValue: "DENY");

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Matches(@"frame-ancestors\s*'none'", securityPolicy);
        }

        [Fact]
        public async Task NoStoreCacheControl()
        {
            HttpResponseMessage response = await GetResponse();

            HttpAssert.HasHeader("Cache-Control", response, expectedValue: "no-store");
        }

        [Fact]
        public async Task DisablesDefaultSrc()
        {
            HttpResponseMessage response = await GetResponse();

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Matches(@"default-src\s*'none'", securityPolicy);
        }

        [Fact]
        public async Task Sandbox()
        {
            HttpResponseMessage response = await GetResponse();

            string securityPolicy = response.Headers.GetValues("Content-Security-Policy").First();

            Assert.Contains("sandbox", securityPolicy);
        }

        [Fact]
        public async Task DoesNotContainServerInfo()
        {
            HttpResponseMessage response = await GetResponse();

            Assert.False(response.Headers.Contains("Server"));
        }
    }
}