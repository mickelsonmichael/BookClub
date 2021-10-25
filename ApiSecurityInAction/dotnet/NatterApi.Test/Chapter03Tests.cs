using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Test.TestHelpers;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter03Tests
    {
        private readonly WebApplicationFactory<Startup> _factory = new();

        [Fact]
        public void LimitRequests()
        {
            const int requests = 5;
            HttpClient client = _factory.CreateClient();

            Task<HttpResponseMessage>[] responses = Enumerable.Range(1, requests)
                .Select(_ => RequestHelpers.GetResponse(client, ensureStatusCode: false))
                .ToArray();

            Task.WaitAll(responses);

            Assert.Contains(responses, x => x.Result.StatusCode == System.Net.HttpStatusCode.TooManyRequests);
        }
    }
}
