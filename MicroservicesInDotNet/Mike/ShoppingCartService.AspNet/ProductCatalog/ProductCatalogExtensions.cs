using Polly;

namespace ShoppingCartService.ProductCatalog;

public static class ProductCatalogExtensions
{
    public static WebApplicationBuilder AddProductCatalog(this WebApplicationBuilder builder)
    {
        ProductCatalogOptions options = AddAndGetOptions(builder);

        if (options.IsInMemory)
        {
            builder.Services.AddTransient<IProductCatalogStore, InMemoryProductCatalogStore>();
        }
        else
        {
            builder.Services.AddHttpClient<IProductCatalogStore, HttpProductCatalogStore>()
                .AddTransientHttpErrorPolicy(policy =>
                    policy.WaitAndRetryAsync(3, attempt => TimeSpan.FromMilliseconds(100 * Math.Pow(2, attempt))));
        }

        return builder;
    }

    private static ProductCatalogOptions AddAndGetOptions(WebApplicationBuilder builder)
    {
        IConfigurationSection configSection = builder.Configuration.GetSection(ProductCatalogOptions.Key);

        builder.Services.Configure<ProductCatalogOptions>(configSection);

        ProductCatalogOptions options = new();

        configSection.Bind(options);

        return options;
    }
}