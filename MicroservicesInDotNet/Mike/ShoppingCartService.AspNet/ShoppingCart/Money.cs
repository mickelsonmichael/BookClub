namespace ShoppingCartService.ShoppingCart;

public readonly record struct Money(
    string Currency,
    decimal Amount
);
