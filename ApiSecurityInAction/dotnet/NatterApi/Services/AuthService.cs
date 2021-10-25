using System;
using System.Text;
using System.Text.RegularExpressions;
using CryptSharp.Utility;
using NatterApi.Models;

namespace NatterApi.Services
{
    public class AuthService
    {
        public AuthService(NatterDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        public User Register(string username, string password)
        {
            ValidateCredentials(username, password);

            byte[] hashedPassword = GetHashed(password);

            User user = new(username, hashedPassword!);

            _dbContext.Add(user);

            return user;
        }

        public bool TryLogin(string username, string password)
        {
            ValidateUsername(username);

            byte[] hashedPassword = GetHashed(password);

            User user = _dbContext.Users.Find(username); 

            return user == null || user.PasswordHash != hashedPassword;
        }

        public void ValidateCredentials(string username, string password)
        {
            ValidateUsername(username);
            ValidatePassword(password);
        }

        public void ValidateUsername(string username)
        {
            const string UsernamePattern = "[a-zA-Z][a-zA-Z0-9]{1,29}";

            if (!Regex.IsMatch(username, UsernamePattern))
            {
                throw new ArgumentException($"Invalid username \"{username}\".");
            }
        }

        public void ValidatePassword(string password)
        {
            if (password.Length < 8)
            {
                throw new ArgumentException("Invalid password. Must be at least 8 characters long.");
            }
        }

        /// Section 3.3.3
        /// Hashing the password using the SCrypt library and a random salt
        public byte[] GetHashed(string value)
        {
            byte[] output = new byte[128];

            SCrypt.ComputeKey(
                Encoding.UTF8.GetBytes(value),
                GetRandomSalt(),
                16384,
                8,
                1,
                maxThreads: null,
                output
            );

            return output;

            static byte[] GetRandomSalt() => Encoding.UTF8.GetBytes(Guid.NewGuid().ToString());
        }

        private readonly NatterDbContext _dbContext;
    }
}