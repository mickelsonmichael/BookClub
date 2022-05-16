namespace FunctionalProgramming.Web.Domain;

public readonly record struct Book(
    string BookId,
    string Title,
    string Author
);
