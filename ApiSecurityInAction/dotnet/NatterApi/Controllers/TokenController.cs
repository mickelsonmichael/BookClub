using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;
using NatterApi.Extensions;
using System;
using System.Linq;
using NatterApi.Filters;

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
        [RequireScope("full_access")]
        public IActionResult Login(
            [FromQuery] string? scope
        )
        {
            string? username = HttpContext.GetNatterUsername();

            if (string.IsNullOrWhiteSpace(username))
            {
                return new UnauthorizedResult();
            }

            DateTime expiry = DateTime.Now.AddMinutes(10);

            scope = string.IsNullOrWhiteSpace(scope) ? DefaultScope : scope;

            Token token = new(expiry, username);

            token.Attributes.Add(("scope", scope));

            string tokenId = _tokenService.CreateToken(HttpContext, token);

            return Created("/sessions", tokenId);
        }

        [HttpDelete("/sessions")]
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
        private string DefaultScope => string.Join(" ", new[]
        {
            "create_space",
            "post_message",
            "read_message",
            "list_messages",
            "delete_message",
            "add_member"
        });
    }
}
