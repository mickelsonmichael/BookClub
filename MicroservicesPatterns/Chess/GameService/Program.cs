using GameService.Game;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton<IGameRepository, InMemoryGameRepository>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.MapGet("/game", GameHandlers.GetGame);

app.MapPost("/game", GameHandlers.CreateGame);

app.MapPost("/game/join", GameHandlers.JoinGame);

app.MapPost("/game/move", GameHandlers.Move);

app.Run();
