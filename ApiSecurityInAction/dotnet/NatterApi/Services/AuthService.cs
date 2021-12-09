using System;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using NatterApi.Models;
using Scrypt;

namespace NatterApi.Services
{
    public class AuthService
    {
        public AuthService(NatterDbContext dbContext, ILogger<AuthService> logger)
        {
            _dbContext = dbContext;
            _logger = logger;
        }

        public User Register(string username, string password)
        {
            ValidateCredentials(username, password);

            /// Section 3.3.3
            /// Hashing the password using SCrypt
            string hashedPassword = _encoder.Encode(password);

            User user = new(username, hashedPassword!);

            _logger.LogInformation("Registering username \"{Username}\".", username);

            _dbContext.Add(user);
            _dbContext.SaveChanges();

            return user;
        }

        public bool TryLogin(string username, string password)
        {
            ValidateUsername(username);

            _logger.LogInformation("Checking credentials for \"{Username}\".", username);

            User user = _dbContext.Users.Find(username);

            return user != null && _encoder.Compare(password, user.PasswordHash);
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

        private readonly NatterDbContext _dbContext;
        private readonly ILogger<AuthService> _logger;
        private readonly ScryptEncoder _encoder = new();
    }
}
