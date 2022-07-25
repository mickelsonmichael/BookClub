using LanguageExt;

namespace ShoppingCartService.ShoppingCart;

public interface IShoppingCartStore
{
    Task<ShoppingCart> Get(int userId);

    Task<Try<Unit>> Save(ShoppingCart shoppingCart);
}
