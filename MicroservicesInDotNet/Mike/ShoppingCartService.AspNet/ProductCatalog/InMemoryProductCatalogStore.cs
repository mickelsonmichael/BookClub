using System.Net;
using LanguageExt;
using ShoppingCartService.ShoppingCart;

namespace ShoppingCartService.ProductCatalog;

using ProductCatalogId = Int32;

public class InMemoryProductCatalogStore : IProductCatalogStore
{
    public Task<Validation<Error, ShoppingCartItem>> Get(int productId)
        => _db.TryGetValue(productId)
            .ToValidation(new Error(HttpStatusCode.NotFound, $"Product {productId} could not be found."))
            .AsTask();

    public Task<Validation<Error, IEnumerable<ShoppingCartItem>>> Get(IEnumerable<int> productIds)
    {
        (IEnumerable<Error> errors, IEnumerable<ShoppingCartItem> items) = productIds
            .Map(productId =>
                _db.TryGetValue(productId)
                    .ToEither(new Error(HttpStatusCode.NotFound, $"Product {productId} could not be found."))
            ).Partition();

        return (
            errors.Any()
            ? Validation<Error, IEnumerable<ShoppingCartItem>>.Fail(errors.ToSeq())
            : Validation<Error, IEnumerable<ShoppingCartItem>>.Success(items)
        ).AsTask();
    }

    private readonly IDictionary<ProductCatalogId, ShoppingCartItem> _db
        = new Dictionary<ProductCatalogId, ShoppingCartItem>
        {
            [1] = new ShoppingCartItem(
                ProductCatalogId: 1,
                ProductName: "Cat tree",
                Description: "Made for scratching.",
                Price: new Money("$", 45m)
            ),
            [2] = new ShoppingCartItem(
                ProductCatalogId: 2,
                ProductName: "Laser pointer",
                Description: "Point it at the ground and watch them lose their minds.",
                Price: new Money("$", 12m)
            )
        };
}
