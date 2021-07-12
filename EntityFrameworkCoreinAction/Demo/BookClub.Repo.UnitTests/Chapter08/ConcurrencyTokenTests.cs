using Microsoft.EntityFrameworkCore;
using System.Linq;
using System.Transactions;
using Xunit;

namespace Repo.UnitTests
{
    public class ConcurrencyTokenTests : IClassFixture<BookClubDbFixture>
    {
        private readonly BookClubDbFixture _fixture;

        public ConcurrencyTokenTests(BookClubDbFixture fixture)
        {
            this._fixture = fixture;
        }

        [Fact]
        public void ConcurrentCheck_ThrowsException()
        {
            using var context = _fixture.OpenContext();
            using var transaction = context.Database.BeginTransaction();

            var book = context.Books.First();

            context.Database.ExecuteSqlInterpolated($@"
                 UPDATE Books
                 SET Edition = 'TESTEDITION'
                 WHERE BookId = {book.BookId};
             ");

            book.Name = "Updated Name";

            Assert.Throws<DbUpdateConcurrencyException>(() => context.SaveChanges());
        }

        [Fact]
        public void ConcurrentCheck_AllowsUpdate()
        {
            using var context = _fixture.OpenContext();
            using var transaction = context.Database.BeginTransaction();

            var book = context.Books.First();

            book.Edition = "Updated Edition";

            Assert.Equal(1, context.SaveChanges());

            // UPDATE "Books" SET "Edition" = @p0
            // WHERE "BookId" = @p1 AND "Edition" = @p2;
            // SELECT changes();
        }
    }
}