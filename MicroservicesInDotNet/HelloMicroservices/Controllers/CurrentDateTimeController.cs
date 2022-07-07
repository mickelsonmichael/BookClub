using Microsoft.AspNetCore.Mvc;

namespace HelloMicroservices.Controllers;

public class CurrentDateTimeController : ControllerBase
{
    [HttpGet("/")]
    public DateTime Get() => DateTime.UtcNow;
}
