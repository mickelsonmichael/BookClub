using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Debug;
using System;
using System.Data.Common;

namespace Repo.UnitTests
{
    public class BookClubDbFixture : IDisposable
    {
        private static readonly object _lock = new object();
        private static bool _initialized;
        private static readonly ILoggerFactory _loggerFactory
            = LoggerFactory.Create(builder => builder.AddDebug());

        public SqliteConnection Connection { get; }

        public BookClubDbFixture()
        {
            Connection = new SqliteConnection("Filename=:memory:");
            Connection.Open();

            SeedDatabase();
        }

        public void Dispose() => Connection.Dispose();

        private void SeedDatabase()
        {
            lock (_lock)
            {
                if (!_initialized)
                {
                    using var context = OpenContext();

                    context.Database.EnsureDeleted();
                    context.Database.EnsureCreated();

                    DbCreator.SeedDatabase(context);

                    _initialized = true;
                }
            }
        }

        public BookClubDbContext OpenContext(DbTransaction transaction = null)
        {
            var options = new DbContextOptionsBuilder<BookClubDbContext>()
                    .UseLoggerFactory(_loggerFactory)
                    .UseSqlite(Connection);

            var context = new BookClubDbContext(
                options.Options
            );

            if (transaction != null)
            {
                context.Database.UseTransaction(transaction);
            }

            return context;
        }
    }
}
