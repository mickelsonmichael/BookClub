using System.ComponentModel.DataAnnotations;

namespace ShoppingCartService.ProductCatalog;

public class ProductCatalogOptions
{
    public const string Key = "ProductCatalog";
    public const string InMemoryValue = "InMemory";

    public bool IsInMemory => ProductCatalogUrl == InMemoryValue;

    [Required(AllowEmptyStrings = false)]
    public string ProductCatalogUrl { get; init; } = InMemoryValue;
}
