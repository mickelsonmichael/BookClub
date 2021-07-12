using Microsoft.EntityFrameworkCore;
using Repo.Models;
using Repo.ValueGenerators;

namespace Repo.Configs
{
    public class AuthorConfig : IEntityTypeConfiguration<Author>
    {
        public void Configure(Microsoft.EntityFrameworkCore.Metadata.Builders.EntityTypeBuilder<Author> builder)
        {
            builder.HasKey(a => a.AuthorId);

            builder.Property(a => a.AuthorId)
                .ValueGeneratedOnAdd();

            builder.Property(a => a.FriendlyId)
                .HasValueGenerator<FriendlyIdGenerator>();
        }
    }
}