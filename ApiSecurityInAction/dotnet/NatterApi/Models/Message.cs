using System;

namespace NatterApi.Models
{
    public class Message
    {
        public Guid MessageId { get; init; }
        public string? Author { get; init; }
        public DateTime? MessageTime { get; init; }
        public string? MessageText { get; init; }
    }
}