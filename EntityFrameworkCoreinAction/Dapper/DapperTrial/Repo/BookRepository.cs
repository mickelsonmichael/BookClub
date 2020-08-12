using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.IO;
using System.Linq;
using Dapper;
using DapperTrial.Models;

namespace DapperTrial.Repo
{
    public class BookRepository
    {
        public static string DbFile => $"{Environment.CurrentDirectory}\\BookDb.Sqlite";

        public static SQLiteConnection DbConnection()
            => new SQLiteConnection($"Data Source={DbFile}");

        public BookRepository()
        {
            if (!File.Exists(DbFile))
            {
                Console.WriteLine("Creating Database...");
                using var connection = DbConnection();
                connection.Open();

                connection.Execute(
                    @"CREATE TABLE Books
                    (
                        BookId INTEGER PRIMARY KEY AUTOINCREMENT,
                        Title varchar(250) NOT NULL,
                        Author varchar(200) NOT NULL
                    )"
                );

                connection.Execute(
                    @"INSERT INTO Books
                        (Title, Author)
                        VALUES
                        ('C# in Depth', 'Jon Skeet')
                    "
                );
            }
        }

        public IEnumerable<Book> GetBooks()
        {
            using var connection = DbConnection();
            connection.Open();

            return connection.Query<Book>(
                @"SELECT BookId, Title, Author
                    FROM Books"
            ) ?? Enumerable.Empty<Book>();
        }

        public void Cleanup()
        {
            if (File.Exists(DbFile))
                File.Delete(DbFile);
        }
    }
}