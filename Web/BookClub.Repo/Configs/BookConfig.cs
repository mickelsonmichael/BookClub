using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Repo.Models;
using System;

namespace Repo.Configs
{
    public class BookConfig : IEntityTypeConfiguration<Book>
    {
        public void Configure(EntityTypeBuilder<Book> builder)
        {
            builder.Property(x => x.BookId)
                .ValueGeneratedOnAdd();

            builder.Property(b => b.CreatedDate)
                .HasDefaultValueSql("datetime('now')");
            //.HasDefaultValue(DateTime.Now); // This won't work. The value will be set on context configuration, not during the add
        }
    }
}