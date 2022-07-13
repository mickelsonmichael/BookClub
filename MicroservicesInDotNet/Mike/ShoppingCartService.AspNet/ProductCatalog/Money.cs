namespace ShoppingCartService.ProductCatalog;

public readonly record struct Money(
    string Currency,
    decimal Amount
);
