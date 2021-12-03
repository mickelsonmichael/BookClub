using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using NatterApi.Extensions;
using NatterApi.Models;

namespace NatterApi.Filters
{
    /// <summary>
    /// <para>
    /// Section 3.6.1 - Enforcing authentication | An ActionFilterAttribute will allow us to decorate any
    /// action or controller to require the method or class be restricted to only authenticated users.
    /// </para>
    /// <para>
    /// Section 8.1 - Additionally, users can be granted access to a space by their associated groups
    /// </para>
    /// <para>
    /// Section 8.2.3 - Permissions are now assigned from roles and are provided to the HttpContext via
    /// the <see cref="LookupPermissionsAttribute"/>.
    /// </para>
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

            ICollection<string> groups = httpContext.Items["groups"] as ICollection<string> ?? new List<string>();
            int? spaceId = httpContext.GetSpaceId();

            if (_accessLevels.Count == 0 || spaceId == null)
            {
                return; // no permissions needed, proceed
            }

            string perms = GetPermissions(httpContext);

            if (perms == null || !HasPermissions(perms))
            {
                Unauthorized(context);
            }
        }

        private void Unauthorized(ActionExecutingContext context)
        {
            context.Result = new UnauthorizedResult();

            context.HttpContext.Response.Headers.Add("WWW-Authenticate", "Basic realm=\"/\", charset=\"UTF-8\"");
        }

        private static string GetPermissions(HttpContext context)
        {
            return context.Items["perms"] as string ?? string.Empty;
        }

        private bool HasPermissions(string permissions)
        {
            return HasAllPermissions() || CanDelete() || CanRead() || CanWrite();

            bool HasAllPermissions() => _accessLevels.Contains(AccessLevel.All)
                    && permissions.Contains('d')
                    && permissions.Contains('r')
                    && permissions.Contains('w');

            bool CanDelete() => _accessLevels.Contains(AccessLevel.Delete) 
                && permissions.Contains('d');

            bool CanRead() => _accessLevels.Contains(AccessLevel.Read)
                && permissions.Contains('r');

            bool CanWrite() => _accessLevels.Contains(AccessLevel.Write)
                && permissions.Contains('w');
        }

        private readonly ISet<string> _accessLevels;
    }
}
