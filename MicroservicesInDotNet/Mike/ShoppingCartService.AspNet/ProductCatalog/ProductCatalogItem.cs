namespace ShoppingCartService.ProductCatalog;

public readonly record struct ProductCatalogItem(
    int ProductCatalogId,
    string ProductName,
    string Description,
    Money Price
)
{
    public bool Equals(ProductCatalogItem? obj)
        => obj != null && ProductCatalogId.Equals(obj.Value.ProductCatalogId);

    public override int GetHashCode() => ProductCatalogId.GetHashCode();
}
