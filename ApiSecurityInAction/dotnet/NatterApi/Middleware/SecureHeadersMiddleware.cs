using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace NatterApi.Middleware
{
    public class SecureHeadersMiddleware
    {
        public SecureHeadersMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            RemoveCacheControl(context.Response);

            context.Response.OnStarting(state =>
            {
                HttpContext context = (HttpContext)state;

                EnforceContentTypeEncoding(context.Response);

                DisableXSSHeader(context.Response);

                SetContentTypeOptionsToNoSniff(context.Response);

                DenyFrameOptions(context.Response);

                SetSecurityPolicyHeader(context.Response);

                RemoveServerInfo(context.Response);

                return Task.CompletedTask;
            }, context);

            await _next(context);
        }

        /// Section 2.6.3
        /// We must explicitly set the character encoding to UTF-8
        /// Otherwise, bad actors could utilize UTF-16BE to hijack the user's data
        /// https://portswigger.net/blog/json-hijacking-for-the-modern-web
        private void EnforceContentTypeEncoding(HttpResponse response)
        {
            switch (response.ContentType)
            {
                case "application/json":
                    response.ContentType = "application/json; charset=UTF-8";
                    break;
            }
        }

        /// Section 2.6.1
        /// Disabling the X-SS-Protection header increases security on several browsers.
        /// Most browsers disable this service by default, but for added security we manually disable it via this header.
        private void DisableXSSHeader(HttpResponse response)
        {
            response.Headers.Add("X-XSS-Protection", "0");
        }

        /// Section 2.6.2 (Table 2.1)
        /// This header prevents the browser from automatically determining
        /// the content type from the shape of the data.
        private void SetContentTypeOptionsToNoSniff(HttpResponse response)
        {
            response.Headers["X-Content-Type-Options"] = "nosniff";
        }

        /// Section 2.6.2 (Table 2.1)
        /// Prevent the response from being rendered into an iframe, which can lead
        /// to a number of common hacks.
        /// This has been replaced by the Content-Security-Policy header
        /// but is still worth setting for older browsers.
        private void DenyFrameOptions(HttpResponse response)
        {
            response.Headers["X-Frame-Options"] = "DENY";
        }

        /// Section 2.6.2 (Table 2.1)
        /// While caching is often a useful feature, it's better to
        /// make it an opt-in feature. This header can be added early
        /// in the middleware pipeline and removed by an endpoint later.
        private void RemoveCacheControl(HttpResponse response)
        {
            response.Headers["Cache-Control"] = "no-store";
        }

        /// Section 2.6.2 
        /// default-src=none prevents the response from loading script resources
        /// frame-ancestors=none is the replacement for X-Frame-Options (see above)
        /// sandbox disables scripts from being executed
        private void SetSecurityPolicyHeader(HttpResponse response)
        {
            response.Headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; sandbox";
        }

        /// Section 2.6
        /// By default, Kestrel will respond with its information,
        /// this is information leak we don't want, so we remove that header.
        private void RemoveServerInfo(HttpResponse response)
        {
            response.Headers.Remove("Server");
        }

        private readonly RequestDelegate _next;
    }
}
