namespace NatterApi.Services.TokenStore
{
    /// <summary>
    /// 6.4 - Using types for secure API design.
    /// A Confidential token service ensures that the token state is kept a secret.
    /// </summary>
    public interface IConfidentialTokenService : ITokenService
    {
    }
}
