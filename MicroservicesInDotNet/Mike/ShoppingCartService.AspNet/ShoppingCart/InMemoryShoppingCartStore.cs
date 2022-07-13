namespace ShoppingCartService.ShoppingCart;

using LanguageExt;
using static LanguageExt.Prelude;

public class InMemoryShoppingCartStore : IShoppingCartStore
{
    public Task<ShoppingCart> Get(int userId)
        => _db.TryGetValue(userId)
            .Match(
                Some: cart => cart,
                None: new ShoppingCart(userId)
            ).AsTask();

    public Try<ShoppingCart> Save(ShoppingCart shoppingCart)
        => Try(_db[shoppingCart.UserId] = shoppingCart);

    private readonly IDictionary<int, ShoppingCart> _db = new Dictionary<int, ShoppingCart>();
}
