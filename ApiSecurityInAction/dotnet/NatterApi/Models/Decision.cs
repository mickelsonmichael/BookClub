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
    )
    {
        // 8.3.1 Combining decisions
        public static Decision Permitted() => new(IsPermitted: true);
        public static Decision Denied() => new(IsPermitted: false);
    };
}
