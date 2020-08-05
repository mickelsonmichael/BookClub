using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using Repo.Models;

namespace Repo.Configs
{
    public class BookConfig : IEntityTypeConfiguration<Book>
    {
        public void Configure(EntityTypeBuilder<Book> builder)
        {
            builder.Property(x => x.BookId)
                .ValueGeneratedOnAdd();

            builder.Property(b => b.CreatedDate)
                .HasDefaultValueSql("GETDATE()");
            //.HasDefaultValue(DateTime.Now); // This won't work. The value will be set on context configuration, not during the add

            // Computed columns example
            builder.Property(b => b.Current)
                .HasComputedColumnSql("CAST(IS NOT NULL [StartedDate] AND IS NULL [CompletedDate] as bit)");
        }
    }
}