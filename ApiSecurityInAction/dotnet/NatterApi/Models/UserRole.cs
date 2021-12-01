using System.ComponentModel.DataAnnotations.Schema;

namespace NatterApi.Models
{
    /// <summary>
    /// 8.2.2 Static Roles
    /// </summary>
    public class UserRole
    {
        // Key (see FluentAPI)
        public int SpaceId { get; }
        // Key (see FluentAPI)
        [ForeignKey(nameof(User))]
        public string Username { get; }
        [ForeignKey(nameof(Role))]
        public string RoleId { get; }

        public User? User { get; private set; }
        public RolePermission? Role { get; private set; }

        public UserRole(int spaceId, string roleId, string username)
        {
            SpaceId = spaceId;
            RoleId = roleId;
            Username = username;
        }
    }
}
