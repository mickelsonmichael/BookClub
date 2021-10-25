using System.Collections.Generic;

namespace NatterApi.Models
{
    public record Space(
        int Id,
        string Name,
        string Owner
    )
    {
        public ICollection<Message>? Messages { get; set; }
    }
}
