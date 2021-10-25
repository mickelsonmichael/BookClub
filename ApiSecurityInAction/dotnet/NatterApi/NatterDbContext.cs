using Microsoft.EntityFrameworkCore;
using NatterApi.Models;

namespace NatterApi
{
    public class NatterDbContext : DbContext
    {
        #nullable disable
        public DbSet<Space> Spaces { get; }
        public DbSet<User> Users { get; }
        #nullable enable

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase("NatterDb");
        }
    }
}