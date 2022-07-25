using System.Net;

namespace ShoppingCartService;

public readonly record struct Error(
    HttpStatusCode StatusCode,
    string Message
);
