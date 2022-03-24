WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

AddServices(builder.Services);

WebApplication app = builder.Build();

BuildPipeline(app);

static void AddServices(IServiceCollection services)
{
    services.AddControllers();
}

static void BuildPipeline(WebApplication app)
{
    app.UseHttpsRedirection();

    app.MapControllers();

    app.Run();
}