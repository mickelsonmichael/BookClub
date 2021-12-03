namespace NatterApi.Models
{
    public record Permission(
        int SpaceId,
        string UsernameOrGroupname,
        string Permissions
    );
}
