namespace ShoppingCartService.ShoppingCart;

using LanguageExt;
using static LanguageExt.Prelude;

public class InMemoryShoppingCartStore : IShoppingCartStore
{
    public InMemoryShoppingCartStore(ILogger<InMemoryShoppingCartStore> logger)
    {
        logger.LogWarning("Using in-memory shopping cart store.");

        _logger = logger;
    }

    public Task<ShoppingCart> Get(int userId)
        => _db.TryGetValue(userId)
            .Match(
                Some: cart => cart,
                None: () =>
                {
                    _logger.LogInformation("Creating new cart for user {UserId}.", userId);

                    return new ShoppingCart(userId);
                }
            ).AsTask();

    public Task<Try<Unit>> Save(ShoppingCart shoppingCart) =>
        (from _ in Try(() => _logger.LogDebug("Saving cart\n{Cart}", shoppingCart))
         from c in Try(_db[shoppingCart.UserId] = shoppingCart)
         select Unit.Default
        ).AsTask();

    private readonly ILogger<InMemoryShoppingCartStore> _logger;
    private readonly IDictionary<int, ShoppingCart> _db = new Dictionary<int, ShoppingCart>();
}
