using Microsoft.AspNetCore.Mvc;

namespace NatterApi.Controllers
{
    [ApiController, Route("/status")]
    [ValidateAntiForgeryToken]
    public class StatusController : ControllerBase
    {
        [HttpGet]
        public IActionResult Index()
        {
            return Ok("app is running");
        }
    }
}
