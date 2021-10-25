using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using Xunit.Sdk;

namespace NatterApi.Test.TestHelpers
{
    public static class HttpAssert
    {
        public static void HasHeader(string headerName, HttpResponseMessage response, string expectedValue = null)
        {
            bool hasHeader = response.Headers.TryGetValues(headerName, out IEnumerable<string> values);

            if (!hasHeader)
            {
                throw new XunitException($"The response does not contain the header. Header name \"{headerName}\".");
            }

            if (values == null)
            {
                return;
            }

            if (values.Count() != 1)
            {
                throw new XunitException($"The \"{headerName}\" header contained more than one value.\n{string.Join(", ", values)}");
            }

            string value = values.First();

            if (value != expectedValue)
            {
                throw new XunitException($"The value of the \"{headerName}\" header does not match \"{expectedValue}\".\nGot \"{value}\" instad.");
            }
        }
    }
}
