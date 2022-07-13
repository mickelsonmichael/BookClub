using LanguageExt;

namespace ShoppingCartService.ShoppingCart;

public interface IShoppingCartStore
{
    public OptionAsync<ShoppingCart> Get(int userId);

    public TryAsync<ShoppingCart> Save(ShoppingCart shoppingCart);
}
