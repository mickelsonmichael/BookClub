using BookClub.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Repo;
using Repo.Models;
using System;
using System.Diagnostics;
using System.Linq;

namespace BookClub.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index([FromServices] BookClubDbContext context)
            => View(context.Books.Include(b => b.Author));

        public IActionResult AddBook([FromServices] BookClubDbContext context)
        {
            var newBook = new Book
            {
                Name = DateTime.Now.ToString(),
                Author = context.Authors.First()
            };

            context.Add(newBook);
            context.SaveChanges();

            return RedirectToAction(nameof(Index));
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
