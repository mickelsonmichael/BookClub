namespace ShoppingCartService.ShoppingCart;

using System.Linq;
using LanguageExt;
using ShoppingCartService.EventFeed;

public readonly record struct ShoppingCart
{
    public int UserId { get; }
    public IEnumerable<ShoppingCartItem> Items { get; init; } = new HashSet<ShoppingCartItem>();

    public ShoppingCart(int userId) => UserId = userId;

    public ShoppingCart AddItem(ShoppingCartItem item, IEventStore eventStore)
    {
        eventStore.Raise("ShoppingCartItemAdded", new { UserId, item });

        return this with { Items = Items.Append(item) };
    }

    public ShoppingCart AddItems(IEnumerable<ShoppingCartItem> items, IEventStore eventStore) =>
        items.Aggregate(this, (cart, item) => cart.AddItem(item, eventStore));

    public ShoppingCart RemoveItem(int productId, IEventStore eventStore)
    {
        eventStore.Raise("ShoppingCartItemRemoved", new { UserId, productId });

        return this with { Items = Items.ExceptBy(new[] { productId }, keySelector: i => i.ProductCatalogId) };
    }

    public ShoppingCart RemoveItems(IEnumerable<int> productIds, IEventStore eventStore) =>
        productIds.Aggregate(this, (cart, productId) => cart.RemoveItem(productId, eventStore));
}
