using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using NatterApi.Test.TestHelpers;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter03Tests : TestWithKestrel
    {
        [Fact]
        public void LimitRequests()
        {
            const int requests = 5;

            Task<HttpResponseMessage>[] responses = Enumerable.Range(1, requests)
                .Select(i => GetResponse(ensureStatusCode: false))
                .ToArray();

            Task.WaitAll(responses);

            Assert.True(responses.Any(x => x.Result.StatusCode == System.Net.HttpStatusCode.TooManyRequests));
        }
    }
}
