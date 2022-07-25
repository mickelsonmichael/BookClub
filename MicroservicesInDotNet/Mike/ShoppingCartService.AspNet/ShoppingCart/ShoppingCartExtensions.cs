namespace ShoppingCartService.ShoppingCart;

public static class ShoppingCartExtensions
{
    public static WebApplicationBuilder AddShoppingCarts(this WebApplicationBuilder builder)
    {
        builder.Services.AddSingleton<IShoppingCartStore, InMemoryShoppingCartStore>();

        return builder;
    }
}
