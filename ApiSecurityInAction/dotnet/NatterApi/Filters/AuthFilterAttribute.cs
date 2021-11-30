using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
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

            ICollection<string> perms = GetPermissions(httpContext, spaceId.Value, username, groups);

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

        private static ICollection<string> GetPermissions(HttpContext context, int spaceId, string username, ICollection<string> groups)
        {
            IServiceProvider services = context.RequestServices;

            NatterDbContext dbContext = services.GetRequiredService<NatterDbContext>();

            return dbContext.Permissions
                .Where(p => p.SpaceId == spaceId && (
                    p.UsernameOrGroupname == username || groups.Contains(p.UsernameOrGroupname)
                ))
                .Select(p => p.Permissions)
                .ToList();
        }

        private bool HasPermissions(ICollection<string> permissions)
        {
            return HasAllPermissions() || CanDelete() || CanRead() || CanWrite();

            bool HasAllPermissions() => _accessLevels.Contains(AccessLevel.All)
                    && permissions.Any(p => p.Contains('d'))
                    && permissions.Any(p => p.Contains('r'))
                    && permissions.Any(p => p.Contains('w'));

            bool CanDelete() => _accessLevels.Contains(AccessLevel.Delete) 
                && permissions.Any(p => p.Contains('d'));

            bool CanRead() => _accessLevels.Contains(AccessLevel.Read)
                && permissions.Any(p => p.Contains('r'));

            bool CanWrite() => _accessLevels.Contains(AccessLevel.Write)
                && permissions.Any(p => p.Contains('w'));
        }

        private readonly ISet<string> _accessLevels;
    }
}
