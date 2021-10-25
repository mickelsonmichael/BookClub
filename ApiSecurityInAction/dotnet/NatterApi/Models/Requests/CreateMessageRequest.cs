using System;
using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models.Requests
{
    public record CreateMessageRequest(
        [Required] string Author,
        [Required] string Message
    )
    {
        public Message ToMessage() => new(
            0,
            Author,
            DateTime.Now,
            Message
        );
    }
}
