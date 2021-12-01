using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.DependencyInjection;
using NatterApi.Extensions;

namespace NatterApi.Filters
{
    public class LookupPermissionsAttribute : ActionFilterAttribute
    {
        public override void OnActionExecuting(ActionExecutingContext context)
        {
            string? username = context.HttpContext.GetNatterUsername();

            if (context.ModelState["spaceId"]?.RawValue is string spaceId && username != null)
            {
                NatterDbContext dbContext = context.HttpContext.RequestServices.GetRequiredService<NatterDbContext>();

                string? permissions = dbContext.UserRoles.Find(spaceId, username)?.Role?.Permissions;

                context.HttpContext.Items["perms"] = permissions ?? string.Empty;
            }
        }
    }
}
