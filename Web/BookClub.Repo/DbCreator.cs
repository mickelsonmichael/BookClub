using System.Linq;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Repo.Models;

namespace Repo
{
    public static class DbCreator
    {
        private static Book[] SeedBooks = new Book[]
        {
            new Book { Name = "C# in Depth", Edition = "Fourth", Author = "Jon Skeet", Complete = true, Image = "https://csharpindepth.com/images/Cover.png" },
            new Book { Name = "Entity Framework Core in Action", Author = "Jon P Smith", Edition = "First", Current = true, Image = "https://images.manning.com/360/480/resize/book/2/2cd7852-84a5-44f5-a05b-451d68478e31/Smith-EFC-HI.png" },
            new Book { Name = "ASP.NET Core in Action", Author = "Andrew Lock", Edition = "First", Complete = true, Image = "https://images.manning.com/360/480/resize/book/3/a3544c6-0057-465a-a17e-d1fbd7f61b80/Lock-ANCore-HI.png" }
        };

        public static DbContextOptionsBuilder CreateDb(this DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite(CreateConnection());

            using var context = new BookClubDbContext(
                (DbContextOptions<BookClubDbContext>)optionsBuilder.Options
            );

            context.Database.EnsureCreated();

            if (!context.Books.Any())
            {
                SeedDatabase(context);
            }

            return optionsBuilder;
        }

        private static SqliteConnection CreateConnection()
        {
            // create the database in-memory
            var connection = new SqliteConnection("Filename=:memory:");

            connection.Open();

            return connection;
        }

        private static void SeedDatabase(BookClubDbContext context)
        {
            context.Books.AddRange(SeedBooks);
            context.SaveChanges();
        }
    }
}