using Microsoft.EntityFrameworkCore;
using Repo.Configs;
using Repo.Models;

namespace Repo
{
    public class BookClubDbContext : DbContext
    {
        public BookClubDbContext(DbContextOptions<BookClubDbContext> options) : base(options)
        {
            // no further configuration needed
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.ApplyConfiguration(new BookConfig());
        }

        public DbSet<Book> Books { get; set; }
    }
}
