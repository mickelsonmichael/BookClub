namespace NatterApi.Models
{
    /// <summary>
    /// 8.3 Attribute-based access control
    /// </summary>
    /// <param name="IsPermitted">
    /// `true` if the decision is to allow the request,
    /// `false` otherwise.
    /// </param>
    public class Decision
    {
        // 8.3.1 Combining decisions
        public bool IsPermitted { get; private set; } = true; // permitted by default

        public void Deny() => IsPermitted = false;
    };
}
