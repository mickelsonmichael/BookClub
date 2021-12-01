using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models
{
    public class RolePermission
    {
        [Key]
        public string RoleId { get; }
        public string Permissions { get; }

        public RolePermission(string roleId, string permissions)
        {
            RoleId = roleId;
            Permissions = permissions;
        }
    }
}
