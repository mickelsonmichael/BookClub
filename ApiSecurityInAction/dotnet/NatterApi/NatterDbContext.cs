using System;
using Microsoft.EntityFrameworkCore;
using NatterApi.Models;
using NatterApi.Models.Token;

namespace NatterApi
{
    public class NatterDbContext : DbContext
    {
        #nullable disable
        public DbSet<Space> Spaces { get; private set; }
        public DbSet<User> Users { get; private set; }
        public DbSet<AuditMessage> AuditLog { get; private set; }
        public DbSet<Permission> Permissions { get; private set; }
        public DbSet<Session> Sessions { get; private set; }
        #nullable enable

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase($"NatterDb-{Guid.NewGuid()}");
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Permission>()
                .HasKey(p => new { p.SpaceId, p.Username });

        }
    }
}
