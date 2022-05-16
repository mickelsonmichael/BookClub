using System.ComponentModel.DataAnnotations;

namespace FunctionalProgramming.Web.Events;

public readonly record struct CreateBook(
    [Required, MinLength(3)]
    string Title,
    [Required, MinLength(1)]
    string Author
);
