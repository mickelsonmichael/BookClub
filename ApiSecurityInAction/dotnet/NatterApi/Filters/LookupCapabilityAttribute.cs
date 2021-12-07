using System.Linq;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;

namespace NatterApi.Filters
{
    /// <summary>
    /// 9.2.2 Using capability URIs in the Natter API
    /// </summary>
    public class LookupCapabilityAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext context)
        {
            string? tokenId = (string?)context.HttpContext.Request.Query["access_token"];

            if (string.IsNullOrWhiteSpace(tokenId))
            {
                return;
            }

            DatabaseTokenService tokenService = context.HttpContext.RequestServices.GetRequiredService<DatabaseTokenService>();

            Token? token = tokenService.ReadToken(context.HttpContext, tokenId);

            if (token == null)
            {
                return;
            }

            string? path = token.GetAttribute("path");
            string requestPath = context.HttpContext.Request.Path;

            if (path == requestPath)
            {
                context.HttpContext.Items["perms"] = token.GetAttribute("perms");
            }
        }
    }
}
