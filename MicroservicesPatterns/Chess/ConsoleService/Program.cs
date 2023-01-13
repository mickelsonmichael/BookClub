
using ConsoleService.Games;

ApiGameRepository gameRepository = new();

GameController gameController = new(gameRepository);

await gameController.Play();
