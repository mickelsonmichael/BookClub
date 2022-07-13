namespace ShoppingCartService.ShoppingCart;

using LanguageExt;
using ShoppingCartService.ProductCatalog;

public readonly record struct ShoppingCart
{
    public int UserId { get; }
    public IEnumerable<ProductCatalogItem> Items { get; init; } = new HashSet<ProductCatalogItem>();

    public ShoppingCart(int userId) => UserId = userId;

    public ShoppingCart AddItem(ProductCatalogItem item) =>
        this with { Items = Items.Append(item) };

    public ShoppingCart AddItems(IEnumerable<ProductCatalogItem> items) =>
        items.Aggregate(this, (cart, item) => cart.AddItem(item));
}
