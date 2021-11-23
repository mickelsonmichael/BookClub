using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;
using NatterApi.Extensions;
using System;
using System.Linq;

namespace NatterApi.Controllers
{
    [ApiController]
    public class TokenController : ControllerBase
    {
        public TokenController(ISecureTokenService tokenService)
        {
            _tokenService = tokenService;
        }

        [HttpPost("/sessions")]
        public IActionResult Login()
        {
            string? username = HttpContext.GetNatterUsername();

            if (string.IsNullOrWhiteSpace(username))
            {
                return new UnauthorizedResult();
            }

            DateTime expiry = DateTime.Now.AddMinutes(10);

            Token token = new(expiry, username);

            string tokenId = _tokenService.CreateToken(HttpContext, token);

            return Created("/sessions", tokenId);
        }

        [HttpGet("/sessions")]
        public IActionResult Logout()
        {
            string? tokenId = Request.Headers["Authorization"].FirstOrDefault();

            if (tokenId?.StartsWith("Bearer") != true)
            {
                throw new ArgumentException("Missing token header");
            }

            tokenId = tokenId["Bearer ".Length..];

            _tokenService.DeleteToken(HttpContext, tokenId);

            return Ok();
        }

        private readonly ISecureTokenService _tokenService;
    }
}
