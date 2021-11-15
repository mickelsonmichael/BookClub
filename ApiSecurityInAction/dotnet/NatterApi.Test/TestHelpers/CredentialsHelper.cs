using System;
using System.Net.Http.Headers;
using System.Text;

namespace NatterApi.Test.TestHelpers
{
    public static class CredentialsHelper
    {
        public static AuthenticationHeaderValue GetCredentials(string username, string password)
        {
            byte[] data = Encoding.UTF8.GetBytes($"{username}:{password}");

            string base64Encoded = Convert.ToBase64String(data);

            return new AuthenticationHeaderValue("Basic", base64Encoded);
        }
    }
}
