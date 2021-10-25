using System.ComponentModel.DataAnnotations;
using System.Security.Claims;
using System.Text.Json.Serialization;

namespace NatterApi.Models
{
    public class User
    {
        [Key]
        public string Username { get; set; }
        [JsonIgnore] // prevent accidental hash leakage
        public string PasswordHash { get; set; }

        public User(string username, string passwordHash)
        {
            Username = username;
            PasswordHash = passwordHash;
        }
    }
}
