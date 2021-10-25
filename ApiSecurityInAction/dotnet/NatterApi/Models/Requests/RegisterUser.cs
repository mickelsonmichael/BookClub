using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models.Requests
{
    public record RegisterUser(
        [Required(AllowEmptyStrings = false), MinLength(3)] string Username,
        [Required(AllowEmptyStrings = false), MinLength(8)] string Password
    );
}
