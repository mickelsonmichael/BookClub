using Xunit;
using System;
using Repo.UnitTests;
using Repo.Models;

namespace Repo.UnitTests.Chapter08
{
    public class HasGeneratedColumnSqlTests : IClassFixture<BookClubDbFixture>
    {
        private readonly BookClubDbFixture _fixture;

        public HasGeneratedColumnSqlTests(BookClubDbFixture fixture)
        {
            this._fixture = fixture;
        }

        // [Fact]
        // public void CurrentBookColumn_IsTrue_WhenStartDateAndNoEndDate()
        // {
        //     using var transaction = _fixture.Connection.BeginTransaction();
        //     using var context = _fixture.OpenContext(transaction);

        //     var author = new Author();

        //     context.Authors.Add(author);
        //     context.SaveChanges();

        //     var book = new Book
        //     {
        //         Name = "Test Book",
        //         Author = author,
        //         CompletedDate = null,
        //         StartDate = DateTime.Now
        //     };

        //     context.Books.Add(book);
        //     context.SaveChanges();

        //     Assert.True(book.Current);
        // }
    }
}