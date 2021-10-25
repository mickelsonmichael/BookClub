using Microsoft.EntityFrameworkCore;
using NatterApi.Models;

namespace NatterApi
{
    public class NatterDbContext : DbContext
    {
        public DbSet<Space> Spaces { get; }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase("NatterDb");
        }
    }
}