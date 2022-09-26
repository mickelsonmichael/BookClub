var builder = WebApplication.CreateBuilder(args);

// Configure Services

builder.Services.AddControllers();

// Configure
var app = builder.Build();

app.UseHttpsRedirection() // Redirect all HTTP requests to HTTPS
    .UseRouting()
    .UseEndpoints(endpoints => endpoints.MapControllers());

app.Run();
