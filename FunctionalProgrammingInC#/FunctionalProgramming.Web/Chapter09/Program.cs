// 9.5.3 Mapping functions to API endpoint

// WebApplication app = WebApplication.Create();

// app.MapGet("/", () => new { status = "running" });

// app.MapPost("/transfers", HandlerFactory.CreateTransferReqestHandler(app.Configuration));

// await app.RunAsync();

// WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

// AddServices(builder.Services, builder.Configuration);

// WebApplication app = builder.Build();

// BuildPipeline(app);

// static void AddServices(IServiceCollection services, IConfiguration config)
// {
//     services.AddControllers();

//     // 9.4.1 Types as documentation
//     services.AddSingleton<ConnectionString>(config.GetConnectionString("Database"));
// }

// static void BuildPipeline(WebApplication app)
// {
//     app.UseHttpsRedirection();

//     app.MapControllers();

//     app.Run();
// }
