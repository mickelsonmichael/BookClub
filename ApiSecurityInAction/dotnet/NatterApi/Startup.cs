using AspNetCoreRateLimit;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;
using NatterApi.Extensions;
using NatterApi.Filters;
using NatterApi.Middleware;
using NatterApi.Services;
using NatterApi.Services.TokenStore;
using System;

namespace NatterApi
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddRateLimiting();

            services.AddDbContext<NatterDbContext>(ServiceLifetime.Singleton);

            services.AddScoped<AuthService>();
            services.AddScoped<ITokenService, EncryptedTokenService>();
            services.AddScoped<ValidateTokenFilterAttribute>();

            services.AddDistributedMemoryCache();
            services.AddSession(options =>
            {
                options.Cookie.Name = "NatterCookie";
                options.IdleTimeout = TimeSpan.FromMinutes(10);
                options.Cookie.HttpOnly = true;
                options.Cookie.IsEssential = false;
            });

            services.AddControllers();

            services.AddSwaggerGen(c => c.SwaggerDoc("v1", new OpenApiInfo { Title = "NatterApi", Version = "v1" }));

            services.AddHostedService<TokenJanitor>();
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            app.UseIpRateLimiting();

            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "NatterApi v1"));
            }

            app.UseHttpsRedirection();

            app.UseMiddleware<NatterCorsMiddleware>();

            app.UseRouting();

            app.UseStaticFiles();

            app.UseSession();

            app.UseMiddleware<SecureHeadersMiddleware>();

            app.UseMiddleware<AuthMiddleware>();

            app.UseMiddleware<AuditMiddleware>();

            app.UseEndpoints(endpoints => endpoints.MapControllers().RequireCors(CorsExtensions.CorsPolicyName));
        }
    }
}
