using Microsoft.AspNetCore.Http;
using NatterApi.Services.TokenStore;
using NatterApi.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Session;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Options;
using System.Text;
using System.Security.Cryptography;

namespace NatterApi.Middleware
{
    public class SessionMiddleware
    {
        private static readonly RandomNumberGenerator CryptoRandom = RandomNumberGenerator.Create();
        private const int SessionKeyLength = 36;
        private static readonly Func<bool> ReturnTrue = () => true;
        public SessionMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(HttpContext context, ITokenService tokenService, SessionOptions options, ISessionStore sessionStore)
        {
            var cookieValue = context.Request.Cookies[options.Cookie.Name];
            var sessionKey = cookieValue;
            var isNewSessionKey = false;
            Func<bool> tryEstablishSession = ReturnTrue;
            if (sessionKey == null || sessionKey.Length != SessionKeyLength)
            {
                var guidBytes = new byte[16];
                CryptoRandom.GetBytes(guidBytes);
                sessionKey = new Guid(guidBytes).ToString();
                cookieValue = Convert.ToBase64String(Encoding.ASCII.GetBytes(sessionKey));
                var establisher = new SessionEstablisher(context, cookieValue, options);
                tryEstablishSession = establisher.TryEstablishSession;
                isNewSessionKey = true;
            }
                

            var sessionFeature = new SessionFeature();
            sessionFeature.Session = sessionStore.Create(sessionKey, options.IdleTimeout, options.IOTimeout, tryEstablishSession, isNewSessionKey);
            context.Features.Set<ISessionFeature>(sessionFeature);

            context.Response.OnStarting(state =>
            {
                HttpContext context = (HttpContext)state;

                var cookieOptions = options.Cookie.Build(context);

                context.Response.Cookies.Append(options.Cookie.Name, cookieValue, cookieOptions);

                return Task.CompletedTask;
            }, context);




            await _next(context);
        }

        private class SessionEstablisher
        {
            private readonly HttpContext _context;
            private readonly string _cookieValue;
            private readonly SessionOptions _options;
            private bool _shouldEstablishSession;

            public SessionEstablisher(HttpContext context, string cookieValue, SessionOptions options)
            {
                _context = context;
                _cookieValue = cookieValue;
                _options = options;
                context.Response.OnStarting(OnStartingCallback, state: this);
            }

            private static Task OnStartingCallback(object state)
            {
                var establisher = (SessionEstablisher)state;
                if (establisher._shouldEstablishSession)
                {
                    establisher.SetCookie();
                }
                return Task.FromResult(0);
            }

            private void SetCookie()
            {
                var cookieOptions = _options.Cookie.Build(_context);

                _context.Response.Cookies.Append(_options.Cookie.Name, _cookieValue, cookieOptions);

                _context.Response.Headers["Cache-Control"] = "no-cache";
                _context.Response.Headers["Pragma"] = "no-cache";
                _context.Response.Headers["Expires"] = "-1";
            }

            // Returns true if the session has already been established, or if it still can be because the response has not been sent.
            internal bool TryEstablishSession()
            {
                return (_shouldEstablishSession |= !_context.Response.HasStarted);
            }
        }

        private readonly RequestDelegate _next;
    }
}
