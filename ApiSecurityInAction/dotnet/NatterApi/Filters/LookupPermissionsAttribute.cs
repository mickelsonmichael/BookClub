using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;
using NatterApi.Models;

namespace NatterApi.Filters
{
    public class LookupPermissionsAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext context)
        {
            string? username = context.HttpContext.GetNatterUsername();

            if (context.ModelState["spaceId"]?.RawValue is string s
                && int.TryParse(s, out int spaceId) 
                && username != null)
            {
                NatterDbContext dbContext = context.HttpContext.RequestServices.GetRequiredService<NatterDbContext>();

                UserRole userRole = dbContext.UserRoles.Find(spaceId, username);

                RolePermission role = dbContext.RolePermissions.Find(userRole.RoleId);

                context.HttpContext.Items["perms"] = role.Permissions;
            }
        }
    }
}
