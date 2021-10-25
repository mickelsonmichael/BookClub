using Microsoft.EntityFrameworkCore;
using NatterApi.Models;

namespace NatterApi
{
    public class NatterDbContext : DbContext
    {
        #nullable disable
        public DbSet<Space> Spaces { get; private set; }
        public DbSet<User> Users { get; private set; }
        public DbSet<Message> Messages { get; private set; }
        #nullable enable

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase("NatterDb");
        }
    }
}
