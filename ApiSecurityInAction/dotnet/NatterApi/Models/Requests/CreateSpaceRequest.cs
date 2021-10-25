using System;

namespace NatterApi.Models.Requests
{
    public record CreateSpaceRequest(
        string Name,
        string Owner
    )
    {
        public Space CreateSpace() => new()
        {
            Id = Guid.NewGuid(),
            Name = Name,
            Owner = Owner
        };
    }
}
