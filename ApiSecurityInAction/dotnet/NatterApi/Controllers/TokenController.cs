using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Filters;
using NatterApi.Models;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;
using NatterApi.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NatterApi.Services;

namespace NatterApi.Controllers
{
    [ApiController]
    public class TokenController : ControllerBase
    {
        public TokenController(ITokenService tokenService)
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

            if (tokenId == null || !tokenId.StartsWith("Bearer"))
            {
                throw new ArgumentException("Missing token header");
            }

            tokenId = tokenId.Substring("Bearer ".Length);

            _tokenService.DeleteToken(HttpContext, tokenId);

            return Ok();
        }

        private readonly ITokenService _tokenService;
    }
}
