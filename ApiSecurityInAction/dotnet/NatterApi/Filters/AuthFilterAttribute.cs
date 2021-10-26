using System;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;

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
        public AuthFilterAttribute(string accessLevel = AccessLevel.None)
        {
            _accessLevel = accessLevel;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            HttpContext httpContext = context.HttpContext;
            string? username = httpContext.GetNatterUsername();

            if (username == null)
            {
                Unauthorized(context);

                return;
            }

            int? spaceId = httpContext.GetSpaceId();

            if (_accessLevel == AccessLevel.None || spaceId == null)
            {
                return;
            }

            string? perms = GetPermissionString(httpContext, spaceId.Value, username);

            if (perms == null || !HasPermissions(perms))
            {
                Unauthorized(context);
            }
        }

        private void Unauthorized(ActionExecutingContext context)
        {
            context.Result = new UnauthorizedResult();

            context.HttpContext.Response.Headers.Add("WWW-Authenticate", "Basic realm=\"/\", charset=\"UFT-8\"");
        }

        private string? GetPermissionString(HttpContext context, int spaceId, string username)
        {
            IServiceProvider services = context.RequestServices;

            NatterDbContext dbContext = services.GetRequiredService<NatterDbContext>();

            Permission? permission = dbContext.Permissions.Find(spaceId, username);

            return permission?.Permissions;
        }

        private bool HasPermissions(string permission)
        {
            return (_accessLevel == AccessLevel.Delete && permission.Contains("d"))
                || (_accessLevel == AccessLevel.Read && permission.Contains("r"))
                || (_accessLevel == AccessLevel.Write && permission.Contains("w"));
        }

        private readonly string _accessLevel;
    }
}
