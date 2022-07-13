using LanguageExt;

namespace ShoppingCartService.ProductCatalog;

using ProductCatalogId = System.Int32;

public class InMemoryProductCatalogStore : IProductCatalogStore
{
    public Task<Validation<Error, ProductCatalogItem>> Get(int productId)
        => _db.TryGetValue(productId)
            .ToValidation(new Error($"Product {productId} could not be found."))
            .AsTask();

    public Task<Validation<Error, IEnumerable<ProductCatalogItem>>> Get(IEnumerable<int> productIds)
    {
        (IEnumerable<Error> errors, IEnumerable<ProductCatalogItem> items) = productIds
            .Map(productId =>
                _db.TryGetValue(productId)
                    .ToEither(new Error($"Product {productId} could not be found."))
            ).Partition();

        return (
            errors.Any()
            ? Validation<Error, IEnumerable<ProductCatalogItem>>.Fail(errors.ToSeq())
            : Validation<Error, IEnumerable<ProductCatalogItem>>.Success(items)
        ).AsTask();
    }

    private readonly IDictionary<ProductCatalogId, ProductCatalogItem> _db
        = new Dictionary<ProductCatalogId, ProductCatalogItem>
        {
            [1] = new ProductCatalogItem(
                ProductCatalogId: 1,
                ProductName: "Cat tree",
                Description: "Made for scratching.",
                Price: new Money("$", 45m)
            ),
            [2] = new ProductCatalogItem(
                ProductCatalogId: 2,
                ProductName: "Laser pointer",
                Description: "Point it at the ground and watch them lose their minds.",
                Price: new Money("$", 12m)
            )
        };
}
