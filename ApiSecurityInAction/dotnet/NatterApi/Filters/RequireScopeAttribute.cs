using System;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace NatterApi.Filters
{
    /// <summary>
    /// <para>Section 7.1.1 - Adding scoped tokens</para>
    /// <para>
    /// This filter will perform several checks to determine whether
    /// or not the required scopes are granted to the current
    /// <see ref="Token" />.
    /// </para>
    /// </summary>
    [AttributeUsage(
        AttributeTargets.Class | AttributeTargets.Method,
        AllowMultiple = true,
        Inherited = true
    )]
    public class RequireScopeAttribute : ActionFilterAttribute
    {
        public RequireScopeAttribute(string requiredScope)
        {
            _requiredScope = requiredScope;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            string? scope = context.HttpContext.Items["scope"] as string;

            if (string.IsNullOrWhiteSpace(scope))
            {
                // the user authenticated using basic auth
                // since having the password would allow creation of
                // any scope, we can treat it like a maximal token
                return;
            }

            string[] scopes = scope.Split(" ");

            if (!scopes.Contains(_requiredScope))
            {
                context.HttpContext.Response.Headers["WWW-Authenticate"]
                    = @$"Bearer error=""insufficient_scope"",scope=""{_requiredScope}""";

                context.Result = new UnauthorizedResult();
            }
        }

        private readonly string _requiredScope;
    }
}