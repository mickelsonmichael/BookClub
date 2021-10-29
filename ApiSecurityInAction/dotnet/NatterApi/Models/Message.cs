using System;
using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models
{
    public record Message(
        int MessageId,
        string Author,
        DateTime MessageTime,
        string MessageText
    );
}
