using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Test.TestHelpers;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter06Tests
    {
        private readonly WebApplicationFactory<Startup> _factory = new();
        private HttpClient HttpClient() => _factory.CreateClient();

        [Fact]
        public async Task LoginWithToken()
        {
            SessionHelper session = new(HttpClient());

            session.Register("test_user", "test_password")
                .Login();

            Assert.NotNull(session.Token);
            
            (HttpResponseMessage response, string spaceId) = await session.CreateSpace("test_space", "test_user");

            Assert.Equal(HttpStatusCode.Created, response.StatusCode);
            Assert.NotNull(spaceId);
        }
    }
}
