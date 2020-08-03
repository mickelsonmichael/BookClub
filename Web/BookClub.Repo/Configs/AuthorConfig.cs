using Microsoft.EntityFrameworkCore;
using Repo.Models;

namespace Repo.Configs
{
    public class AuthorConfig : IEntityTypeConfiguration<Author>
    {
        public void Configure(Microsoft.EntityFrameworkCore.Metadata.Builders.EntityTypeBuilder<Author> builder)
        {
            builder.HasKey(a => a.AuthorId);

            builder.Property(a => a.AuthorId)
                .ValueGeneratedOnAdd();
        }
    }
}