namespace ShoppingCartService.ShoppingCart;

public readonly record struct ShoppingCartItem(
    int ProductCatalogId,
    string ProductName,
    string Description,
    Money Price
)
{
    public bool Equals(ShoppingCartItem? obj)
        => obj != null && ProductCatalogId.Equals(obj.Value.ProductCatalogId);

    public override int GetHashCode() => ProductCatalogId.GetHashCode();
}
