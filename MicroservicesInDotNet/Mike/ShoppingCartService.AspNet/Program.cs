using ShoppingCartService.ShoppingCart;
using ShoppingCartService.ProductCatalog;

var builder = WebApplication.CreateBuilder(args);

// Configure Services

builder.Services.AddTransient<IShoppingCartStore, InMemoryShoppingCartStore>();
builder.Services.AddTransient<IProductCatalogStore, InMemoryProductCatalogStore>();

// Configure
var app = builder.Build();

app.UseHttpsRedirection(); // Redirect all HTTP requests to HTTPS

app.MapGet("/shoppingcart/{userId:int}", ShoppingCartHandler.GetCart);

app.MapPost("/shoppingcart/{userId:int}/items", ShoppingCartHandler.PostItems);

app.Run();
