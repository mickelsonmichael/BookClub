using Microsoft.Extensions.Configuration;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;

namespace BookClub.Repo.UnitTests
{
    public static class TestConfiguration
    {
        public static IConfigurationRoot GetConfiguration()
        {
            return new ConfigurationBuilder()
                .SetBasePath(GetTopLevelDirectory())
                .AddJsonFile("appsettings.json")
                .Build();
        }

        // See http://codebuckets.com/2017/10/19/getting-the-root-directory-path-for-net-core-applications/
        private static string GetTopLevelDirectory()
        {
            var root = Assembly.GetExecutingAssembly().CodeBase;
            var dir = Path.GetDirectoryName(root);
            Regex appPathMatcher = new Regex(@"(?<!fil)[A-Za-z]:\\+[\S\s]*?(?=\\+bin)");
            return appPathMatcher.Match(dir).Value;
        }
    }
}
