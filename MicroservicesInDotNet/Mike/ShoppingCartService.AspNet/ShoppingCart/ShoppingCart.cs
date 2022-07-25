namespace ShoppingCartService.ShoppingCart;

using System.Linq;
using LanguageExt;

public readonly record struct ShoppingCart
{
    public int UserId { get; }
    public IEnumerable<ShoppingCartItem> Items { get; init; } = new HashSet<ShoppingCartItem>();

    public ShoppingCart(int userId) => UserId = userId;

    public ShoppingCart AddItem(ShoppingCartItem item) =>
        this with { Items = Items.Append(item) };

    public ShoppingCart AddItems(IEnumerable<ShoppingCartItem> items) =>
        items.Aggregate(this, (cart, item) => cart.AddItem(item));

    public ShoppingCart RemoveItems(IEnumerable<int> productIds) =>
        this with { Items = Items.ExceptBy(productIds, keySelector: i => i.ProductCatalogId) };
}
