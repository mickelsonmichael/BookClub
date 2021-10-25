using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using NatterApi.Extensions;

namespace NatterApi.Filters
{
    /// <summary>
    /// Section 3.6.1 - Enforcing authentication
    /// An ActionFilterAttribute will allow us to decorate any
    /// action or controller to require the method or class be restricted
    /// to only authenticated users.
    /// </summary>
    public class AuthFilterAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext context)
        {
            HttpContext httpContext = context.HttpContext;

            if (httpContext.GetNatterUsername() == null)
            {
                context.Result = new UnauthorizedResult();

                context.HttpContext.Response.Headers.Add("WWW-Authenticate", "Basic realm=\"/\", charset=\"UFT-8\"");
            }
        }
    }
}
