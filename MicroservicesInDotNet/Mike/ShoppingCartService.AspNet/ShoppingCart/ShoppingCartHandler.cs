using LanguageExt;
using Microsoft.AspNetCore.Mvc;
using ShoppingCartService.ProductCatalog;


namespace ShoppingCartService.ShoppingCart;

public static class ShoppingCartHandler
{
    public static Task<IResult> GetCart(
        [FromServices] IShoppingCartStore shoppingCartStore,
        int userId
    ) => shoppingCartStore.Get(userId).Map(c => Results.Ok(c));

    public async static Task<IResult> PostItems(
        [FromServices] IShoppingCartStore shoppingCartStore,
        [FromServices] IProductCatalogStore productCatalogStore,
        int userId,
        [FromBody] int[] productIds
    )
    {
        Option<ShoppingCart> cart = await shoppingCartStore.Get(userId);
        Validation<Error, IEnumerable<ProductCatalogItem>> products = await productCatalogStore.Get(productIds);

        ShoppingCart c = new ShoppingCart(12345);



        var r = productCatalogStore.Get(productIds)
            .Map(
                either => either
                    .Map(items => ((ShoppingCart)cart).AddItems(items))
                    .Map(cart => shoppingCartStore.Save(cart))
                    .Match(
                        cart => Results.Ok(cart),
                        err => Results.StatusCode(500)
                    )
        );

        return products
            .MatchAsync(
            SuccAsync: async p => (await shoppingCartStore.Save(
                    p.Fold(
                        state: (ShoppingCart)cart,
                        (newCart, item) => newCart.AddItem(item)
                    )
                )).Match(
                    Succ: c => Results.Ok(c),
                    Fail: err => Results.StatusCode(500)
                ),
            Fail: (errors) => Results.NotFound(errors));
    }
}
