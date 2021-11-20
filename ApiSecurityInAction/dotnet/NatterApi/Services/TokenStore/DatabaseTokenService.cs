using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
using System;
using System.Security.Cryptography;
using System.Text.Json;

namespace NatterApi.Services.TokenStore
{
    /// <summary>
    /// 5.2.1 - Instead of relying on cookies, we store our tokens in the database
    /// See Listing 5.8 for Java implementation.
    /// </summary>
    public class DatabaseTokenService : ITokenService
    {
        public DatabaseTokenService(NatterDbContext dbContext, ILogger<DatabaseTokenService> logger)
        {
            _dbContext = dbContext;
            _logger = logger;

            logger.LogInformation("Utilizing the {TokenStoreType}.", nameof(DatabaseTokenService));
        }

        public string CreateToken(HttpContext context, Token token)
        {
            _logger.LogDebug("Creating new token.\n{Token}", token);

            string tokenId = CreateUniqueId();

            // JSON serialization of the attributes property is handled by EF Core
            // see the ModelBuilder method inside the NatterDbContext for implementation

            Token toStore = token with { Id = tokenId };

            _dbContext.Add(toStore);
            _dbContext.SaveChanges();

            return tokenId;
        }

        private string CreateUniqueId()
        {
            var bytes = new byte[20];

            RandomNumberGenerator.Fill(bytes);

            return Convert.ToBase64String(bytes);
        }

        public void DeleteToken(HttpContext context, string tokenId)
        {
            _logger.LogDebug("Deleting token <{TokenId}>.", tokenId);

            Token? token = _dbContext.Tokens.Find(tokenId);

            if (token != null)
            {
                _dbContext.Remove(token);
                _dbContext.SaveChanges();
            }
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            _logger.LogDebug("Reading token <{TokenId}>.", tokenId);

            return _dbContext.Tokens.Find(tokenId);
        }

        private readonly NatterDbContext _dbContext;
        private ILogger<DatabaseTokenService> _logger;
    }
}
