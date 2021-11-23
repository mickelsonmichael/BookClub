namespace NatterApi.Services.TokenStore
{
    /// <summary>
    /// 6.4 - Using types for secure API design.
    /// A Secure token service ensures the token cannot be tampered with or faked,
    /// and that the token state is kept secret.
    /// <seealso ref="IAuthenticatedTokenService" />
    /// <seealso ref="IConfidentialTokenService" />
    /// </summary>
    public interface ISecureTokenService
        : IAuthenticatedTokenService, IConfidentialTokenService
    {
    }
}
