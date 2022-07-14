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

    public Task<Try<Unit>> Save(ShoppingCart shoppingCart)
    {
        _db[shoppingCart.UserId] = shoppingCart;

        return Task.FromResult(Try(Unit.Default));
    }

    private readonly IDictionary<int, ShoppingCart> _db = new Dictionary<int, ShoppingCart>();
}
