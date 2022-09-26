using LanguageExt;
using ShoppingCartService.ShoppingCart;

namespace ShoppingCartService.ProductCatalog;

public interface IProductCatalogStore
{
    public Task<Validation<Error, IEnumerable<ShoppingCartItem>>> Get(IEnumerable<int> productIds);
}
