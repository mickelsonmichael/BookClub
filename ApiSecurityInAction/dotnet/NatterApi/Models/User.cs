using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace NatterApi.Models
{
    public class User
    {
        [Key]
        public string Username { get; set; }
        [JsonIgnore] // prevent accidental hash leakage
        public byte[] PasswordHash { get; set; }

        public User(string username, byte[] passwordHash)
        {
            Username = username;
            PasswordHash = passwordHash;
        }
    }
}
