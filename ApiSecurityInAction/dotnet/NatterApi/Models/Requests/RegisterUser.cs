using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models.Requests
{
    public record RegisterUser(
        [Required(AllowEmptyStrings = false)] string Username,
        [Required(AllowEmptyStrings = false)] string Password
    );
}
