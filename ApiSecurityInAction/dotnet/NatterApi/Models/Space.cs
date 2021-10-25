using System.Collections.Generic;

namespace NatterApi.Models
{
    public record Space(
        int Id,
        string Name,
        string Owner
    )
    {
        public IEnumerable<Message>? Messages { get; set; }
    }
}
