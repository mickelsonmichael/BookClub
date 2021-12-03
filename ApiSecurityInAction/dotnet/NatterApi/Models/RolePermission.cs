using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models
{
    public class RolePermission
    {
        public string RoleId { get; private set; }
        public string Permissions { get; private set; }

        public RolePermission(string roleId, string permissions)
        {
            RoleId = roleId;
            Permissions = permissions;
        }
    }
}
