using Dapper;
using LanguageExt;
using static FunctionalProgramming.Notes.Chapter09.Examples.ConnectionHelper;

using SqlTemplate = System.String;

namespace FunctionalProgramming.Notes.Chapter09.Examples;

public static class ConnectionStringExt
{
    public static Func<object, IEnumerable<T>> Get<T>(
        this ConnectionString connectionString,
        SqlTemplate sql
    ) => param => Connect(
        connectionString,
        connection => connection.Query<T>(sql, param)
    );

    public static Try<int> Try(this ConnectionString connectionString, SqlTemplate sql)
        => () => Connect(connectionString, conn => conn.Execute(sql));
}
