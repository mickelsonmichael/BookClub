using System.Text.Json.Serialization;

namespace NatterApi.Models
{
    public class User
    {
        public string Username { get; set; }
        [JsonIgnore]
        public byte[] PasswordHash { get; set; }

        public User(string username, byte[] passwordHash)
        {
            Username = username;
            PasswordHash = passwordHash;
        }
    }
}
