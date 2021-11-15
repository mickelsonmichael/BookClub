using System;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models.Token;
using NatterApi.Services;
using NatterApi.Services.TokenStore;

namespace NatterApi.Filters
{
    /// <summary>
    /// Section 4.4.3
    /// </summary>
    public class ValidateTokenFilterAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext context)
        {
            HttpContext httpContext = context.HttpContext;

            var tokenService = httpContext.RequestServices.GetRequiredService<ITokenService>();

            string? csrfToken = CSRFService.GetToken(httpContext);

            if (string.IsNullOrWhiteSpace(csrfToken))
            {
                return;
            }

            Token? token = tokenService.ReadToken(context.HttpContext, csrfToken);

            if (token == null)
            {
                return;
            }

            if (token.Expiration > DateTime.Now)
            {
                context.HttpContext.SetNatterUsername(token.Username);

                foreach ((string key, string value) in token.Attributes)
                {
                    context.HttpContext.Items[key] = value;
                }
            }
        }
    }
}
