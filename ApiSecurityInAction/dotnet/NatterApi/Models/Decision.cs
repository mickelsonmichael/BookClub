namespace NatterApi.Models
{
    /// <summary>
    /// 8.3 Attribute-based access control
    /// </summary>
    /// <param name="IsPermitted">
    /// `true` if the decision is to allow the request,
    /// `false` otherwise.
    /// </param>
    public record Decision(
        bool IsPermitted    
    );
}
