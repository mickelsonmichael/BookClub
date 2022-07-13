using LanguageExt;

namespace ShoppingCartService.ProductCatalog;

public interface IProductCatalogStore
{
    public Task<Either<Error, IEnumerable<ProductCatalogItem>>> Get(IEnumerable<int> productIds);
}
