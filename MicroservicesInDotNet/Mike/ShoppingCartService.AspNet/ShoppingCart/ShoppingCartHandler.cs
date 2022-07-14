using LanguageExt;
using Microsoft.AspNetCore.Mvc;
using ShoppingCartService.ProductCatalog;


namespace ShoppingCartService.ShoppingCart;

public static class ShoppingCartHandler
{
    public static Task<IResult> GetCart(
        [FromServices] IShoppingCartStore shoppingCartStore,
        int userId
    ) => shoppingCartStore.Get(userId)
                        .Map(c => Results.Ok(c))
                        .AsTask();

    public static Task<IResult> PostItems(
        [FromServices] IShoppingCartStore shoppingCartStore,
        [FromServices] IProductCatalogStore productCatalogStore,
        int userId,
        [FromBody] int[] productIds
    ) =>
    productCatalogStore.Get(productIds)
        .Bind(validation =>
            validation.Bind<Task<Try<Unit>>>(
                products =>
                    shoppingCartStore.Get(userId)
                        .Map(c => c.AddItems(products))
                        .Bind(c => shoppingCartStore.Save(c))
            ).Traverse(f => f)
        ).Map(
            validation => validation.Match(
                Succ: t => t.Match(
                    Succ: _ => Results.Ok(),
                    Fail: Results.StatusCode(500)
                ),
                Fail: errs => Results.NotFound(errs)
            )
        );
}
