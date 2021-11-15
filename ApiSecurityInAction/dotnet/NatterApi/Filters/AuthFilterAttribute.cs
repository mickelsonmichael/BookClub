using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;
using NatterApi.Services;

namespace NatterApi.Filters
{
    /// <summary>
    /// Section 3.6.1 - Enforcing authentication
    /// An ActionFilterAttribute will allow us to decorate any
    /// action or controller to require the method or class be restricted
    /// to only authenticated users.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = true, Inherited = true)]
    public class AuthFilterAttribute : ActionFilterAttribute
    {
        public AuthFilterAttribute(params string[]? accessLevels)
        {
            _accessLevels = accessLevels?.ToHashSet() ?? new HashSet<string>();
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

            if (_accessLevels.Count == 0 || spaceId == null)
            {
                return; // no permissions needed, proceed
            }

            string? perms = GetPermissionString(httpContext, spaceId.Value, username);

            if (perms == null || !HasPermissions(perms) || !SessionFixationService.VerifyFixationToken(httpContext))
            {
                Unauthorized(context);
            }
        }

        private void Unauthorized(ActionExecutingContext context)
        {
            context.Result = new UnauthorizedResult();

            context.HttpContext.Response.Headers.Add("WWW-Authenticate", "Basic realm=\"/\", charset=\"UTF-8\"");
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
            return HasAllPermissions() || CanDelete() || CanRead() || CanWrite();

            bool HasAllPermissions() => _accessLevels.Contains(AccessLevel.All) && permission.Contains("d") && permission.Contains("r") && permission.Contains("w");
            bool CanDelete() => _accessLevels.Contains(AccessLevel.Delete) && permission.Contains("d");
            bool CanRead() => _accessLevels.Contains(AccessLevel.Read) && permission.Contains("r");
            bool CanWrite() => _accessLevels.Contains(AccessLevel.Write) && permission.Contains("w");
        }



        private readonly ISet<string> _accessLevels;
    }
}
