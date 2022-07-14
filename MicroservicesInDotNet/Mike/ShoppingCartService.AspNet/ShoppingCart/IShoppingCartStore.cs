using LanguageExt;

namespace ShoppingCartService.ShoppingCart;

public interface IShoppingCartStore
{
    public Task<ShoppingCart> Get(int userId);

    public Task<Try<Unit>> Save(ShoppingCart shoppingCart);
}
