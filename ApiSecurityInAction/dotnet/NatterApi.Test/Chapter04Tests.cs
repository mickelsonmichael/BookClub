using System;
using System.Collections.Generic;
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
using static NatterApi.Test.TestHelpers.CredentialsHelper;

namespace NatterApi.Test
{
    public class Chapter04Tests
    {
        private readonly WebApplicationFactory<Startup> _factory = new();

        [Fact]
        [Trait("Section", "4.3.1")]
        [Trait("Topic", "Session fixation attacks")]
        public async Task PreventSessionFixation()
        {
            HttpClient attacker = _factory.CreateClient();
            HttpClient victim = _factory.CreateClient();

            // 1. attacker logs in and gets a session cookie
            HttpRequestMessage request = new (HttpMethod.Post, "/sessions");
            request.Headers.Authorization = GetCredentials("attacker", "attack_that_fool");

            HttpResponseMessage response = await attacker.SendAsync(request);
            response.EnsureSuccessStatusCode();

            IEnumerable<string> sessionInfo = response.Headers.GetValues("Set-Cookie");

            Assert.NotEmpty(sessionInfo);
        }
    }
}
