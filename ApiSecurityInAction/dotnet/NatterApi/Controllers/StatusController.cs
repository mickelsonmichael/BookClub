using Microsoft.AspNetCore.Mvc;

namespace NatterApi.Controllers
{
    [ApiController, Route("/status")]
    public class StatusController : ControllerBase
    {
        [HttpGet]
        public IActionResult Index()
        {
            return Ok("app is running");
        }
    }
}
