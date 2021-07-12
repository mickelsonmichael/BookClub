using BookClub.Repo.UnitTests;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
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

        public IConfiguration Configuration { get; }
        public string ConnectionString { get; }

        public BookClubDbFixture()
        {
            Configuration = TestConfiguration.GetConfiguration();
            ConnectionString = Configuration.GetConnectionString("TestConnection");

            SeedDatabase();
        }

        public void Dispose()
        {
            // nothing to dispose
        }

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
                    .UseSqlServer(ConnectionString);

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
