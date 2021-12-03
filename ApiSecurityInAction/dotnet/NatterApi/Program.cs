using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace NatterApi
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args)
                .Build()
                .InitializeDb()
                .Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();

                    /// Section 3.4.1
                    /// Enable HTTPS
                    webBuilder.UseKestrel(kestrel =>
                    {
                        int port = args.Length > 0 ? int.Parse(args[0]) : 4567;

                        kestrel.Listen(IPAddress.Loopback, port, listenOptions
                            => listenOptions.UseHttps("localhost.p12", "changeit"));
                    });
                });

        private static IHost InitializeDb(this IHost host)
        {
            using var scope = host.Services.CreateScope();

            var services = scope.ServiceProvider;
            var context = services.GetRequiredService<NatterDbContext>();

            context.Database.EnsureCreated();
            
            return host;
        }
    }
}
