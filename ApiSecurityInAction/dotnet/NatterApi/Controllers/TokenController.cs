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

        [HttpPost("/sessions"), ValidateTokenFilter]
        public IActionResult Login()
        {
            string? username = HttpContext.GetNatterUsername();

            if (string.IsNullOrWhiteSpace(username))
            {
                return new UnauthorizedResult();
            }
            
            DateTime expiry = DateTime.Now.AddMinutes(10);

            Token token = new(expiry, username);

            string tokenId = _tokenService.CreateToken(Request, token);

            SessionFixationService.CreateFixationToken(HttpContext);

            return Created($"/sessions/{tokenId}", tokenId);
        }


        private readonly ITokenService _tokenService;
    }
}
