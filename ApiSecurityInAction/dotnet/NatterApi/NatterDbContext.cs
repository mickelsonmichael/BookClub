using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.ChangeTracking;
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
        public DbSet<Token> Tokens { get; private set; }
#nullable enable

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase($"NatterDb-{Guid.NewGuid()}");
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Permission>()
                .HasKey(p => new { p.SpaceId, p.Username });

            modelBuilder.Entity<Token>(tokenBuilder =>
            {
                // 5.2.1 - Token attributes should be serialized into the database as JSON instead of rows
                tokenBuilder.Property(t => t.Attributes)
                    .HasConversion(
                        list => JsonSerializer.Serialize(list, _jsonOptions),
                        str => JsonSerializer.Deserialize<List<(string, string)>>(str, _jsonOptions) ?? new List<(string, string)>(),
                        ValueComparer.CreateDefault(typeof(List<(string, string)>), false)
                    );
            });

        }

        private readonly JsonSerializerOptions _jsonOptions = new();
    }
}
