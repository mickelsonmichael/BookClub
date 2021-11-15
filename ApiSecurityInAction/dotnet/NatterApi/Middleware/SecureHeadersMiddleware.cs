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

                if (context.Response.HasStarted)
                {
                    return Task.CompletedTask;
                }

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

        /// <summary>
        /// Section 2.6.3
        /// We must explicitly set the character encoding to UTF-8
        /// Otherwise, bad actors could utilize UTF-16BE to hijack the user's data
        /// https://portswigger.net/blog/json-hijacking-for-the-modern-web
        /// </summary>
        private void EnforceContentTypeEncoding(HttpResponse response)
        {
            switch (response.ContentType)
            {
                case "application/json":
                    response.ContentType = "application/json; charset=UTF-8";
                    break;
            }
        }

        /// <summary>
        /// Section 2.6.1
        /// Disabling the X-SS-Protection header increases security on several browsers.
        /// Most browsers disable this service by default, but for added security we manually disable it via this header.
        /// </summary>
        private void DisableXSSHeader(HttpResponse response)
        {
            response.Headers["X-XSS-Protection"] = "0";
        }

        /// <summary>
        /// Section 2.6.2 (Table 2.1)
        /// This header prevents the browser from automatically determining
        /// the content type from the shape of the data.
        /// </summary>
        private void SetContentTypeOptionsToNoSniff(HttpResponse response)
        {
            response.Headers["X-Content-Type-Options"] = "nosniff";
        }

        /// <summary>
        /// Section 2.6.2 (Table 2.1)
        /// Prevent the response from being rendered into an iframe, which can lead
        /// to a number of common hacks.
        /// This has been replaced by the Content-Security-Policy header
        /// but is still worth setting for older browsers.
        /// </summary>
        private void DenyFrameOptions(HttpResponse response)
        {
            response.Headers["X-Frame-Options"] = "DENY";
        }

        /// <summary>
        /// Section 2.6.2 (Table 2.1)
        /// While caching is often a useful feature, it's better to
        /// make it an opt-in feature. This header can be added early
        /// in the middleware pipeline and removed by an endpoint later.
        /// </summary>
        private void RemoveCacheControl(HttpResponse response)
        {
            response.Headers["Cache-Control"] = "no-store";
        }

        /// <summary>
        /// Section 2.6.2
        /// default-src=none prevents the response from loading script resources
        /// frame-ancestors=none is the replacement for X-Frame-Options (see above)
        /// sandbox disables scripts from being executed
        /// </summary>
        private void SetSecurityPolicyHeader(HttpResponse response)
        {
            response.Headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'; sandbox";
        }

        /// <summary>
        /// Section 2.6
        /// By default, Kestrel will respond with its information,
        /// this is information leak we don't want, so we remove that header.
        /// </summary>
        private void RemoveServerInfo(HttpResponse response)
        {
            response.Headers["Server"] = string.Empty;
        }

        private readonly RequestDelegate _next;
    }
}
