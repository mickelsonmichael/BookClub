using System.Data;
using System.Data.SqlClient;

namespace FunctionalProgramming.Web.Chapter09.Examples;

public static class ConnectionHelper
{
    public static TResult Connect<TResult>(string connectionString, Func<IDbConnection, TResult> f)
    {
        using IDbConnection connection = new SqlConnection(connectionString);

        connection.Open();

        return f(connection);
    }
}
