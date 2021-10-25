using System.Collections.Generic;

namespace NatterApi.Models.Requests
{
    public record CreateSpaceRequest(
        string Name,
        string Owner
    )
    {
        public Space CreateSpace() => new(
            0,
            Name,
            Owner
        );
    }
}
