namespace NatterApi.Models
{
    public record Permission(
        int SpaceId,
        string Username,
        string Permissions
    );
}
