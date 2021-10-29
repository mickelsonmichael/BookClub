using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace NatterApi
{
    public class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args).Build().Run();
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
                        kestrel.Listen(IPAddress.Loopback, 4567, listenOptions
                            => listenOptions.UseHttps("localhost.p12", "changeit"));
                    });
                });
    }
}
