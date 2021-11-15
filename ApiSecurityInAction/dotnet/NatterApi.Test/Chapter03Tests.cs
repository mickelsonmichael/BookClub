using System;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Models.Requests;
using NatterApi.Test.TestHelpers;
using Newtonsoft.Json;
using Xunit;

namespace NatterApi.Test
{
    public class Chapter03Tests
    {
        private readonly WebApplicationFactory<Startup> _factory = new();

        [Fact]
        [Trait("Section", "3.2.1")]
        [Trait("Topic", "Rate limiting")]
        public void LimitRequests()
        {
            const int requests = 15;
            HttpClient client = _factory.CreateClient();

            Task<HttpResponseMessage>[] responses = Enumerable.Range(1, requests)
                .Select(_ => RequestHelpers.GetAsync(client, "/spaces"))
                .ToArray();

            Task.WaitAll(responses);

            Assert.Contains(responses, x => x.Result.StatusCode == System.Net.HttpStatusCode.TooManyRequests);
        }

        [Fact]
        [Trait("Section", "3.3.5")]
        [Trait("Topic", "Authentication")]
        public async Task UserMustRegisterToCreateASpace()
        {
            // Arrange
            HttpClient client = _factory.CreateClient();
            const string username = "tester";
            const string password = "test_password";
            CreateSpaceRequest baseRequest = new("test space", "tester");

            // Act
            HttpResponseMessage badResponse = await RequestHelpers.PostAsync(client, "/spaces", baseRequest, username: null);

            await Task.Delay(1000); // prevent rate limiting errors

            await RequestHelpers.RegisterUser(client, username, password);

            HttpResponseMessage goodResponse = await RequestHelpers.PostAsync(client, "/spaces", baseRequest, username, password);

            // Assert
            Assert.Equal(HttpStatusCode.Unauthorized, badResponse.StatusCode);
            Assert.Equal(HttpStatusCode.Created, goodResponse.StatusCode);
        }


        [Fact]
        [Trait("Section", "3.6.1")]
        [Trait("Topic", "Enforcing authentication")]
        public async Task UnauthorizedRequestReturnsAuthenticateHeader()
        {
            // Arrange
            HttpClient client = _factory.CreateClient();
            CreateSpaceRequest baseRequest = new("test space", "tester");

            // Act
            HttpResponseMessage badResponse = await RequestHelpers.PostAsync(client, "/spaces", baseRequest, username: null);

            // Assert
            HttpAssert.HasHeader("WWW-Authenticate", badResponse, "Basic realm=\"/\", charset=\"UTF-8\"");
        }

        [Fact]
        [Trait("Section", "3.6.5")]
        [Trait("Topic", "Avoiding privilege escalation attacks")]
        public async Task PreventsPrivilegeEscalationAttack()
        {
            // ARRANGE
            HttpClient client = _factory.CreateClient();
            var user1 = (username: "test_user_1", password: "test_password_1");
            var user2 = (username: "test_user_2", password: "test_password_2");
            var badUser = (username: "test_evil", password: "burnt_toast");
            CreateSpaceRequest createRequest = new("test_pace", user1.username);

            // ACT
            // register users
            await RequestHelpers.RegisterUser(client, user1.username, user1.password);
            await RequestHelpers.RegisterUser(client, user2.username, user2.password);

            await Task.Delay(1000); // rate limiting

            await RequestHelpers.RegisterUser(client, badUser.username, badUser.password);

            // create space
            HttpResponseMessage createSpaceResponse = await RequestHelpers.PostAsync(
                client,
                "/spaces",
                new CreateSpaceRequest("test_space", user1.username),
                user1.username,
                user1.password
            );
            createSpaceResponse.EnsureSuccessStatusCode();

            await Task.Delay(1000); // rate limiting

            // create message
            string spaceUrl = createSpaceResponse.Headers.Location.ToString();

            HttpResponseMessage createMessageResponse = await RequestHelpers.GetAsync(
                client,
                spaceUrl,
                user1.username,
                user1.password
            );
            createMessageResponse.EnsureSuccessStatusCode();

            // give user2 permissions
            HttpResponseMessage goodPermissionsResponse = await RequestHelpers.PostAsync(
                client,
                $"{spaceUrl}/members",
                new AddMemberRequest(user2.username, "r"),
                user1.username,
                user1.password
            );
            goodPermissionsResponse.EnsureSuccessStatusCode();

            await Task.Delay(1000); // rate limiting

            // give baduser permissions
            HttpResponseMessage badPermissionsResponse = await RequestHelpers.PostAsync(
                client,
                $"{spaceUrl}/members",
                new AddMemberRequest(badUser.username, "rwd"),
                user2.username,
                user2.password
            );

            // ASSERT
            Assert.Equal(HttpStatusCode.Unauthorized, badPermissionsResponse.StatusCode);
        }
    }
}
