using Microsoft.AspNetCore.Mvc;

namespace FunctionalProgramming.Web.Chapter09.Controllers;

[ApiController]
[Route("")]
public class DefaultController : ControllerBase
{
    public DefaultController(ILogger<DefaultController> logger)
    {
        _logger = logger;
    }

    [HttpGet]
    public IActionResult GetHealth()
    {
        _logger.LogInformation("Health check OK");

        return Ok(new { status = "running" });
    }

    private readonly ILogger<DefaultController> _logger;
}
