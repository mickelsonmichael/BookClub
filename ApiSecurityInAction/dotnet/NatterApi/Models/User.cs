using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Security.Claims;
using System.Text.Json.Serialization;

namespace NatterApi.Models
{
    public class User
    {
        [Key]
        public string Username { get; private set; }
        [JsonIgnore] // prevent accidental hash leakage
        public string PasswordHash { get; private set; }
        [JsonIgnore]
        public ICollection<Group> Groups { get; private set; } = new List<Group>();

        public User(string username, string passwordHash)
        {
            Username = username;
            PasswordHash = passwordHash;
        }
    }
}
