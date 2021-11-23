using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Extensions;
using NatterApi.Models;
using NatterApi.Models.Requests;
using NatterApi.Services;

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


        private readonly AuthService _auth;
    }
}
