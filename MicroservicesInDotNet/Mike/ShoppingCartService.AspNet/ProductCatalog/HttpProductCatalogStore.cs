using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using LanguageExt;
using Microsoft.Extensions.Options;
using ShoppingCartService.ShoppingCart;

namespace ShoppingCartService.ProductCatalog;

using static LanguageExt.Prelude;

using GetItemsValidation = Validation<Error, IEnumerable<ShoppingCartItem>>;

public class HttpProductCatalogStore : IProductCatalogStore
{
    public HttpProductCatalogStore(
        HttpClient client,
        IOptions<ProductCatalogOptions> options,
        ILogger<HttpProductCatalogStore> logger
    )
    {
        logger.LogDebug("Initializing product catalog store with URL {URL}.", options.Value.ProductCatalogUrl);

        client.BaseAddress = new Uri(options.Value.ProductCatalogUrl);

        client.DefaultRequestHeaders
            .Accept
            .Add(new MediaTypeWithQualityHeaderValue("application/json"));

        _client = client;
        _logger = logger;
    }

    public Task<GetItemsValidation> Get(IEnumerable<int> productIds) =>
        RequestItems(productIds)
            .Bind(ConvertToItems);

    private Task<HttpResponseMessage> RequestItems(IEnumerable<int> productCatalogIds)
        => Id(string.Join(separator: ",", productCatalogIds))
            .Map(joined => string.Format("?productIds=[{0}]", joined))
            .Map(_client.GetAsync)
            .Value;

    private static Task<GetItemsValidation> ConvertToItems(HttpResponseMessage response) =>
        response.Content.ReadAsStringAsync()
            .Map(result => response.StatusCode switch
            {
                HttpStatusCode.OK => DeserializeItems(result),
                _ => Fail(response.StatusCode, result)
            });

    private static GetItemsValidation DeserializeItems(string body) =>
        Try(
            JsonSerializer.Deserialize<IEnumerable<ProductCatalogProduct>>(body, _serializerOptions)
            ?? throw new Exception("Deserialized to null")
        )
        .Map(
            items => items.Map(
                p => new ShoppingCartItem(
                    ProductCatalogId: p.ProductId,
                    ProductName: p.ProductName,
                    Description: p.ProductDescription,
                    Price: new(p.Price.Currency, p.Price.Amount)
                )
            )
        )
        .Match(
            Succ: items => GetItemsValidation.Success(items),
            Fail: Fail(HttpStatusCode.InternalServerError, "Unable to retrieve the list of products.")
        );

    private static GetItemsValidation Fail(HttpStatusCode statusCode, string errorMessage)
        => GetItemsValidation.Fail(
            new[]
            {
                new Error(statusCode, errorMessage)
            }.ToSeq());

    private record ProductCatalogProduct(
        int ProductId,
        string ProductName,
        string ProductDescription,
        Money Price
    );

    private readonly HttpClient _client;
    private readonly ILogger<HttpProductCatalogStore> _logger;
    private static readonly JsonSerializerOptions _serializerOptions = new(JsonSerializerDefaults.Web);
}
