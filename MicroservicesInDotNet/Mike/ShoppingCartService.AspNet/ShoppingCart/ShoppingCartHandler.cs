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
        .Bind(valid => valid.Match(
            Succ: products =>
                shoppingCartStore.Get(userId)
                    .Map(c => c.AddItems(products))
                    .Bind(c =>
                        shoppingCartStore.Save(c)
                            .Map(t => t.Match(
                                Succ: _ => Results.Ok(c),
                                Fail: Results.StatusCode(500)
                            ))
                    ),
            Fail: err => Results.NotFound(err).AsTask()
        ));
}
