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

            Assert.True(response.Headers.Contains("X-XSS-Protection"));
            Assert.Equal("0", response.Headers.GetValues("X-XSS-Protection").First());
        }
    }
}