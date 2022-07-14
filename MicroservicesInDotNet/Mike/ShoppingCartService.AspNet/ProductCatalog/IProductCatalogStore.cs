using LanguageExt;

namespace ShoppingCartService.ProductCatalog;

public interface IProductCatalogStore
{
    public Task<Validation<Error, IEnumerable<ProductCatalogItem>>> Get(IEnumerable<int> productIds);
}
