namespace NatterApi.Models.Requests
{
    public record AddMemberRequest(
        string Username,
        string Permissions
    );
}
