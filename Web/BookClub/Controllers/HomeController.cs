using BookClub.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using Repo;

namespace BookClub.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index([FromServices] BookClubDbContext dbContext)
        {
            dbContext.Database.EnsureCreated();
            return View(dbContext.Books);
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
