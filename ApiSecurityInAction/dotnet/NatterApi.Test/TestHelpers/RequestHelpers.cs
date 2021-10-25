using System.Net.Http;
using System.Threading.Tasks;

namespace NatterApi.Test.TestHelpers
{
    public static class RequestHelpers
    {
        public static async Task<HttpResponseMessage> GetResponse(
            HttpClient client,
            HttpRequestMessage request = null,
            bool ensureStatusCode = true
        )
        {
            request ??= GetDefault();

            HttpResponseMessage response = await client.SendAsync(request).ConfigureAwait(false);

            if (ensureStatusCode)
            {
                response.EnsureSuccessStatusCode();
            }

            return response;
        }

        private static HttpRequestMessage GetDefault()
        {
            return new(HttpMethod.Get, "/spaces");
        }
    }
}
