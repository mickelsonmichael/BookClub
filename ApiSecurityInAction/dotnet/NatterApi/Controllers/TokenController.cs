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

namespace NatterApi.Controllers
{
    [ApiController, AuthFilter]
    public class TokenController : ControllerBase
    {
        public TokenController(ITokenService tokenService)
        {
            _tokenService = tokenService;
        }

        [HttpPost("/sessions")]
        [ServiceFilter(typeof(ValidateTokenFilterAttribute))]
        public IActionResult Login()
        {
            string username = HttpContext.GetNatterUsername();
            DateTime expiry = DateTime.Now.AddMinutes(10);

            Token token = new(expiry, username);
            string tokenId = _tokenService.CreateToken(Request, token);

            return Created($"/sessions/{tokenId}", tokenId);
        }


        private readonly ITokenService _tokenService;
    }
}
