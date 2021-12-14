using System;
using System.Linq;
using Macaroons;
using Microsoft.AspNetCore.Http;

namespace NatterApi.Verifiers
{
    public static class SinceVerifier
    {
        public static void AddSinceVerifier(this Verifier verifier, HttpContext context)
        {
            verifier.SatisfyGeneral(packet => Verify(packet, context));
        }

        public static bool Verify(Packet packet, HttpContext httpContext)
        {
            string caveat = packet.Encoding.GetString(packet.Data);

            if (caveat.StartsWith("since > "))
            {
                DateTime minimum = DateTime.Parse(caveat[8..]);
                DateTime requestSince = DateTime.Now.AddDays(-1); // default to one day

                if (httpContext.Request.Query["since"].FirstOrDefault() != null)
                {
                    requestSince = DateTime.Parse(httpContext.Request.Query["since"].First());
                }

                return requestSince > minimum;
            }

            return false;
        }
    }
}