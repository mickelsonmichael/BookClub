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
        public IActionResult Login()
        {
            var tokenId = _tokenService.CreateToken(Request);

            return Created($"/sessions/{tokenId}", tokenId);
        }


        private readonly ITokenService _tokenService;
    }
}
