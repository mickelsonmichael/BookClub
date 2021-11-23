using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using NatterApi.Models.Token;
using System;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

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

        /// <summary>
        /// <para>
        /// 5.3.1 - Hashing database tokens in order to increase security.
        /// Since SHA-256 is a one-way encryption, if our database is hacked,
        /// we will still have secure session tokens.
        /// </para>
        /// </summary>
        public string CreateToken(HttpContext context, Token token)
        {
            _logger.LogDebug("Creating new token.\n{Token}", token);

            string tokenId = CreateUniqueId();

            string hashedTokenId = Hash(tokenId);

            // JSON serialization of the attributes property is handled by EF Core
            // see the ModelBuilder method inside the NatterDbContext for implementation

            Token toStore = token with { Id = hashedTokenId };

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

            string hashedId = Hash(tokenId);

            Token? token = _dbContext.Tokens.Find(hashedId);

            if (token != null)
            {
                _dbContext.Remove(token);
                _dbContext.SaveChanges();
            }
        }

        public Token? ReadToken(HttpContext context, string tokenId)
        {
            _logger.LogDebug("Reading token <{TokenId}>.", tokenId);

            string hashedId = Hash(tokenId);

            return _dbContext.Tokens.Find(hashedId);
        }

        public async Task ClearExpiredTokens()
        {
            _logger.LogDebug("Checking for expired tokens.");

            DateTime now = DateTime.Now;

            IQueryable<Token> expiredTokens = _dbContext.Tokens.Where(token => token.Expiration < now);
            
            int expiredTokenCount = expiredTokens.Count();

            if (expiredTokenCount > 0)
            {
                _logger.LogInformation("Clearing out {Count} expired tokens.", expiredTokenCount);

                _dbContext.RemoveRange(expiredTokens);
                
                await _dbContext.SaveChangesAsync();
            }
        }
        
        private string Hash(string value)
        {
            byte[] unhashedBytes = Encoding.UTF8.GetBytes(value);

            byte[] hashedBytes = SHA256.HashData(unhashedBytes);

            return Convert.ToBase64String(hashedBytes);
        }

        private readonly NatterDbContext _dbContext;
        private ILogger<DatabaseTokenService> _logger;
    }
}
