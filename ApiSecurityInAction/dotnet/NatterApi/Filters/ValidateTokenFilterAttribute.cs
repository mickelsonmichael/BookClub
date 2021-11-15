using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;

namespace NatterApi.Filters
{
    /// <summary>
    /// Section 4.4.3
    /// </summary>
    public class ValidateTokenFilterAttribute : ActionFilterAttribute
    {
        public ValidateTokenFilterAttribute(ITokenService tokenStore)
        {
            _tokenStore = tokenStore;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            HttpContext httpContext = context.HttpContext;
            string? username = httpContext.GetNatterUsername();

            Token? token = _tokenStore.ReadToken(context.HttpContext, null);

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

        private readonly ITokenService _tokenStore;
    }
}
