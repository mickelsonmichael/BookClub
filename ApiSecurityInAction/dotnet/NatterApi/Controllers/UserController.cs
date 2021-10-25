using System;
using System.ComponentModel.DataAnnotations;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models;
using NatterApi.Models.Requests;
using CryptSharp.Utility;
using System.Text;

namespace NatterApi.Controllers
{
    [ApiController, Route("/users")]
    public class UserController : ControllerBase
    {
        private const string UsernamePattern = "[a-zA-Z][a-zA-Z0-9]{1,29}";

        /// Section 3.3.4
        /// Registering a user
        [HttpPost]
        public IActionResult Register(
            [FromBody] RegisterUser registrationInfo
        )
        {
            (string username, string password) = registrationInfo;
            
            if (!Regex.IsMatch(username, UsernamePattern))
            {
                throw new ArgumentException($"Invalid username \"{username}\".");
            }

            byte[] hashedPassword = new byte[128];

            /// Section 3.3.3
            /// Hashing the password using the SCrypt library and a random salt
            SCrypt.ComputeKey(
                Encoding.UTF8.GetBytes(password),
                GetRandomSalt(),
                16384,
                8,
                1,
                maxThreads: null,
                output: hashedPassword
            );

            User user = new(username, hashedPassword!);

            return Created($"/users/{username}", new { username = user.Username });
        }

        private static byte[] GetRandomSalt() => Encoding.UTF8.GetBytes(Guid.NewGuid().ToString());
    }
}
