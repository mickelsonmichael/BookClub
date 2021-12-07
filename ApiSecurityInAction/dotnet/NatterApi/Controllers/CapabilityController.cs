using System;
using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;

namespace NatterApi.Controllers
{
    [ApiController, Route("/capabilities")]
    public class CapabilityController : ControllerBase
    {
        /// <summary>
        /// 9.2.6 Hardening capability URIs
        /// </summary>
        [HttpPost]
        public IActionResult Share(
            [FromBody] CapabilityRequest capReq,
            [FromServices] DatabaseTokenService tokenService
        )
        {
            Uri capUri = new(capReq.Uri);

            // get the value of `access_token=x`            
            string tokenId = capUri.Query[(capUri.Query.IndexOf('=') + 1)..];

            Token? token = tokenService.ReadToken(HttpContext, tokenId);

            if (token == null)
            {
                throw new InvalidOperationException("No token found on request.");
            }

            if (token.GetAttribute("path") != capUri.LocalPath)
            {
                throw new InvalidOperationException($"Incorrect path. {token.GetAttribute("path")} != {capUri.LocalPath}.");
            }

            string tokenPermissions = token.GetAttribute("perms") ?? string.Empty;
            string permissions = capReq.Perms ?? tokenPermissions;

            if (!permissions.Contains(tokenPermissions))
            {
                return Forbid();
            }

            Token newToken = new(token.Expiration, capReq.User);
            newToken.AddAttribute("path", capUri.LocalPath);
            newToken.AddAttribute("perms", permissions);

            string newTokenId = tokenService.CreateToken(HttpContext, newToken);

            UriBuilder ub = new(capReq.Uri);
            ub.Query = $"?access_token={newTokenId}";

            return Ok(new {
                uri = ub.Uri
            });
        }

        public record CapabilityRequest(
            [Required(AllowEmptyStrings = false)]
            string Uri,
            [Required(AllowEmptyStrings = false)]
            string User,
            string? Perms
        );
    }
}
