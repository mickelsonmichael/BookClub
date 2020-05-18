using BookClub.Models;
using Microsoft.AspNetCore.Mvc;
using Repo;
using System.Diagnostics;

namespace BookClub.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index(
            [FromServices] BookClubDbContext dbContext
        ) => View(dbContext.Books);

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
