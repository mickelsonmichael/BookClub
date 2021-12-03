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
        public DbSet<Token> Tokens { get; private set; }
        public DbSet<GroupMember> GroupMembers { get; private set; }
        public DbSet<RolePermission> RolePermissions { get; private set; }
        // 8.2.2 Static Roles
        public DbSet<UserRole> UserRoles { get; private set; }
#nullable enable

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseInMemoryDatabase($"NatterDb-{Guid.NewGuid()}");
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
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

            // 8.2.1
            modelBuilder.Entity<RolePermission>(roleBuilder =>
            {
                roleBuilder.HasKey(role => role.RoleId);

                roleBuilder.HasData(
                    new RolePermission("owner", "rwd"),
                    new RolePermission("moderator", "rd"),
                    new RolePermission("member", "rw"),
                    new RolePermission("observer", "r")
                );
            });

            modelBuilder.Entity<GroupMember>()
                .HasKey(nameof(GroupMember.GroupId), nameof(GroupMember.Username));

            // 8.2.2
            modelBuilder.Entity<UserRole>(ur =>
            {
                ur.HasKey(nameof(UserRole.SpaceId), nameof(UserRole.Username));

                ur.HasOne<RolePermission>()
                    .WithMany()
                    .HasForeignKey(r => r.RoleId);
            });
        }

        private readonly JsonSerializerOptions _jsonOptions = new();
    }
}
