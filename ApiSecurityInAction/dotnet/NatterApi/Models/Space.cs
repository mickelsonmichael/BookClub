using System;
using System.Collections.Generic;

namespace NatterApi.Models
{
    public class Space
    {
        public Guid Id { get; init; }
        public string? Name { get; init; }
        public string? Owner { get; init; }
        public ICollection<Message>? Messages { get; init; }
    }
}
