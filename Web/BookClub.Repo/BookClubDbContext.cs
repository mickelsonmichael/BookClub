using Microsoft.EntityFrameworkCore;
using Repo.Models;

namespace Repo
{
    public class BookClubDbContext : DbContext
    {
        public BookClubDbContext(DbContextOptions<BookClubDbContext> options) : base(options)
        {
            // no further configuraiton needed
        }

        public DbSet<Book> Books { get; set; }
    }
}
