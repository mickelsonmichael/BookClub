using System.Linq;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Repo.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using System.Data;

namespace Repo
{
    public static class DbCreator
    {
        private static SqliteConnection connection;

        public static Author[] SeedAuthors = new Author[]
        {
            new Author { Name = "Jon Skeet" },
            new Author { Name = "Jon P Smith" },
            new Author { Name = "Andrew Lock" }
        };

        public static Book[] SeedBooks = new Book[]
        {
            new Book { Name = "C# in Depth", Edition = "Fourth", Author = SeedAuthors.Single(x => x.Name == "Jon Skeet"), Complete = true, Image = "https://csharpindepth.com/images/Cover.png" },
            new Book { Name = "Entity Framework Core in Action", Author = SeedAuthors.Single(x => x.Name == "Jon P Smith"), Edition = "First", Current = true, Image = "https://images.manning.com/360/480/resize/book/2/2cd7852-84a5-44f5-a05b-451d68478e31/Smith-EFC-HI.png" },
            new Book { Name = "ASP.NET Core in Action", Author = SeedAuthors.Single(x => x.Name == "Andrew Lock"), Edition = "First", Complete = true, Image = "https://images.manning.com/360/480/resize/book/3/a3544c6-0057-465a-a17e-d1fbd7f61b80/Lock-ANCore-HI.png" }
        };

        public static DbContextOptionsBuilder ConfigureDb(this DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseSqlite(CreateConnection());

            return optionsBuilder;
        }

        public static IHost CreateDb(this IHost host)
        {
            using var scope = host.Services.CreateScope();

            var services = scope.ServiceProvider;
            var context = services.GetRequiredService<BookClubDbContext>();

            context.Database.EnsureCreated();

            if (!context.Books.Any())
            {
                SeedDatabase(context);
            }

            return host;
        }

        private static SqliteConnection CreateConnection()
        {
            if (connection == null)
            {
                // create the database in-memory
                connection = new SqliteConnection("Filename=:memory:");
            }

            if (connection.State == ConnectionState.Closed)
            {
                // ensure the connection is open
                connection.Open();
            }


            return connection;
        }

        public static void SeedDatabase(BookClubDbContext context)
        {
            context.Authors.AddRange(SeedAuthors);
            context.SaveChanges();

            context.Books.AddRange(SeedBooks);
            context.SaveChanges();
        }
    }
}