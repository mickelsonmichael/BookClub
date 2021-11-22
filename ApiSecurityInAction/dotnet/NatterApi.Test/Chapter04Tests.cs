using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Models.Requests;
using NatterApi.Test.TestHelpers;
using Xunit;
using static NatterApi.Test.TestHelpers.CredentialsHelper;
using static NatterApi.Test.TestHelpers.RequestHelpers;

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
            HttpClient httpClient = _factory.CreateClient();
            SessionHelper attacker = new(httpClient);
            SessionHelper victim = new(httpClient);

            // ===================================
            // attacker and victim create accounts
            // ===================================
            attacker.Register(username: "attacker", password: "attack_that_fool");
            victim.Register(username: "victim", password: "innocence");

            // =============================================================
            // victim logs in and creates a space the attacker cannot access
            // =============================================================
            victim.Login()
                .CreateSpace("victim_space", "victim", out string spaceId);

            // ==========================================
            // attacker logs in and gets the token
            // ==========================================
            attacker.Login();

            // =============================================================================
            // victim logs in with a malicious link (could include a script tag or something)
            // malicious link includes the attacker's session ID in the request
            // https://owasp.org/www-community/attacks/Session_fixation
            // =============================================================================
            HttpRequestMessage createVictimSession = new(HttpMethod.Post, "/sessions")
            {
                Content = new StringContent(string.Empty, Encoding.UTF8, "application/json")
            };

            createVictimSession.Headers.Authorization = GetBearerCredentials(attacker.Token);

            HttpResponseMessage victimSessionResp = await httpClient.SendAsync(createVictimSession);

            victimSessionResp.EnsureSuccessStatusCode();

            string victimSessionId = await victimSessionResp.Content.ReadAsStringAsync();

            Assert.NotEqual(victimSessionId, attacker.Token); // session ids should be the same

            // =========================================================================
            // attacker should now be able to use his session to view the victim's space
            // =========================================================================
            Assert.Throws<HttpRequestException>(() => attacker.GetSpace(spaceId));
        }
    }
}
