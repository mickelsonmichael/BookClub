using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc.Testing;
using NatterApi.Models.Requests;
using NatterApi.Test.TestHelpers;
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
            WebApplicationFactoryClientOptions doNotHandleCookies = new()
            {
                HandleCookies = false
            };

            HttpClient attacker = _factory.CreateClient(doNotHandleCookies);
            HttpClient victim = _factory.CreateClient(doNotHandleCookies);

            // ===================================
            // attacker and victim create accounts
            // ===================================
            HttpRequestMessage createAttackerAccount = new(HttpMethod.Post, "/users")
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(new { username = "attacker", password = "attack_that_fool" }),
                    Encoding.UTF8,
                    "application/json"
                )
            };

            (await attacker.SendAsync(createAttackerAccount)).EnsureSuccessStatusCode();

            HttpRequestMessage createVictimAccount = new(HttpMethod.Post, "/users")
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(new { username = "victim", password = "innocence" }),
                    Encoding.UTF8,
                    "application/json"
                )
            };

            (await victim.SendAsync(createVictimAccount)).EnsureSuccessStatusCode();

            // =============================================================
            // victim logs in and creates a space the attacker cannot access
            // =============================================================
            HttpRequestMessage victimLogin = new(HttpMethod.Post, "/sessions")
            {
                Content = new StringContent(string.Empty, Encoding.UTF8, "application/json")
            };

            victimLogin.Headers.Authorization = GetCredentials("victim", "innocence");

            HttpResponseMessage victimLoginResponse = await victim.SendAsync(victimLogin);
            string victimSession = GetAuthCookie(victimLoginResponse);

            HttpRequestMessage victimCreateSpace = new(HttpMethod.Post, "/spaces")
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(new CreateSpaceRequest("victim_space", "victim")),
                    Encoding.UTF8,
                  "application/json"
                )
            };

            victimCreateSpace.Headers.Add("Cookie", victimSession);

            HttpResponseMessage createSpaceResponse = await victim.SendAsync(victimCreateSpace);

            string getSpaceUrl = createSpaceResponse.Headers.Location.OriginalString;

            // ==========================================
            // attacker logs in and gets a session cookie
            // ==========================================
            HttpRequestMessage createAttackerSession = new(HttpMethod.Post, "/sessions")
            {
                Content = new StringContent(string.Empty, Encoding.UTF8, "application/json")
            };
            
            createAttackerSession.Headers.Authorization = GetCredentials("attacker", "attack_that_fool");

            HttpResponseMessage attackerSessionResp = await attacker.SendAsync(createAttackerSession);
            attackerSessionResp.EnsureSuccessStatusCode();

            string attackerSessionId = await attackerSessionResp.Content.ReadAsStringAsync();

            string attackerCookie = GetAuthCookie(attackerSessionResp);

            // =============================================================================
            // victim logs in with a malicious link (could include a script tag or something)
            // malicious link includes the attacker's session ID in the request
            // https://owasp.org/www-community/attacks/Session_fixation
            // =============================================================================
            HttpRequestMessage createVictimSession = new(HttpMethod.Post, "/sessions")
            {
                Content = new StringContent(string.Empty, Encoding.UTF8, "application/json")
            };
            createVictimSession.Headers.Authorization = GetCredentials("victim", "innocence");
            createVictimSession.Headers.Add("Cookie", attackerCookie);

            HttpResponseMessage victimSessionResp = await victim.SendAsync(createVictimSession);
            victimSessionResp.EnsureSuccessStatusCode();

            string victimSessionId = await victimSessionResp.Content.ReadAsStringAsync();

            Assert.Equal(victimSessionId, attackerSessionId); // session ids should be the same

            // =========================================================================
            // attacker should now be able to use his session to view the victim's space
            // =========================================================================
            HttpRequestMessage badSpaceRequest = new(HttpMethod.Get, getSpaceUrl);
            badSpaceRequest.Headers.Add("Cookie", attackerCookie);

            HttpResponseMessage badSpaceResponse = await attacker.SendAsync(badSpaceRequest);

            Assert.Equal(HttpStatusCode.Unauthorized, badSpaceResponse.StatusCode);

            static string GetAuthCookie(HttpResponseMessage responseMessage)
            {
                IEnumerable<string> cookies = responseMessage.Headers.GetValues("Set-Cookie");

                string setNatterCookie = cookies.Single(x => x.Contains("NatterCookie"));

                return setNatterCookie.Substring(0, setNatterCookie.IndexOf(";"));
            }
        }
    }
}
