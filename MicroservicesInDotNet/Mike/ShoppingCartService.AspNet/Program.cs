using ShoppingCartService.ShoppingCart;
using ShoppingCartService.ProductCatalog;

var builder = WebApplication.CreateBuilder(args);

// Configure Services

builder.AddShoppingCarts();
builder.AddProductCatalog();

// Configure
var app = builder.Build();

app.UseHttpsRedirection(); // Redirect all HTTP requests to HTTPS

app.MapGet("/shoppingcart/{userId:int}", ShoppingCartHandler.GetCart);

app.MapPost("/shoppingcart/{userId:int}/items", ShoppingCartHandler.PostItems);

app.MapDelete("/shoppingcart/{userId:int}/items", ShoppingCartHandler.DeleteItems);

app.Run();
