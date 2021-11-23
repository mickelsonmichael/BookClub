namespace NatterApi.Services.TokenStore
{
    /// <summary>
    /// 6.4 - Using types for secure API design.
    /// An Authenticated token service ensures the token cannot be tampered with or faked.
    /// </summary>
    public interface IAuthenticatedTokenService : ITokenService
    {
    }
}
