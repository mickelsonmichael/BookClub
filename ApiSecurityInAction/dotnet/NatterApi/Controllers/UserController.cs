using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models;
using NatterApi.Models.Requests;
using NatterApi.Services;
using System;
using System.Collections.Generic;
using System.Security.Claims;

namespace NatterApi.Controllers
{
    [ApiController, Route("/users")]
    public class UserController : ControllerBase
    {
        public UserController(AuthService auth)
        {
            _auth = auth;
        }

        /// <summary>
        /// Section 3.3.4
        /// Registering a user
        /// </summary>
        [HttpPost]
        public IActionResult Register(
            [FromBody] RegisterUser registrationInfo
        )
        {
            (string username, string password) = registrationInfo;

            User user = _auth.Register(username, password);

            return Created($"/users/{username}", new { username = user.Username });
        }

        [HttpPost]
        public IActionResult Login([FromBody] User userdetails)
        {
            if ((!string.IsNullOrWhiteSpace(userdetails.Username)) && (!string.IsNullOrEmpty(userdetails.PasswordHash)))
            {
                if (_auth.TryLogin(userdetails.Username, userdetails.PasswordHash))
                {
                    var claimsIdentity = _auth.SetClaims(userdetails.Username);

                    var authProperties = new AuthenticationProperties
                    {
                        ExpiresUtc = DateTime.Now.AddMinutes(10),
                    };

                    return Ok(HttpContext.SignInAsync(
                    CookieAuthenticationDefaults.AuthenticationScheme,
                    new ClaimsPrincipal(claimsIdentity),
                    authProperties));
                }
            }

            return Forbid();
        }


        private readonly AuthService _auth;
    }
}
