using System;
using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Extensions;
using NatterApi.Extensions;
using NatterApi.Models.Token;
using NatterApi.Services.TokenStore;

namespace NatterApi.Services
{
    /// <summary>
    /// 9.2.2 Using capability URIs in the Natter API
    /// </summary>
    public class CapabilityService
    {
        public CapabilityService(MacaroonTokenService tokenService)
        {
            _tokenService = tokenService;
        }

        public Uri CreateUri(
            HttpContext context,
            string path,
            string permissions,
            TimeSpan expiryDuration
        )
        {
            Token token = new(
                Expiration: DateTime.Now.Add(expiryDuration),
                Username: context.GetNatterUsername() ?? string.Empty
            );

            token.AddAttribute("path", path);
            token.AddAttribute("perms", permissions);

            string tokenId = _tokenService.CreateToken(context, token);

            UriBuilder ub = new(context.Request.GetDisplayUrl());

            ub.Query = $"?access_token={tokenId}";

            return ub.Uri;
        }

        private readonly ISecureTokenService _tokenService;
    }
}
