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
using NatterApi.Options;
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

            // 9.2.5 - Combining capabilities with identity
            // Because we are relying on capabilities for permissions,
            // we now only need a session for auditing and minor access checks
            // like when defining the owner of a space.
            services.AddHttpClient<ISecureTokenService, CookieTokenService>();

            services.AddScoped<DatabaseTokenService>()
                .AddScoped<CapabilityService>();

            services.AddScoped<ValidateTokenFilterAttribute>();

            services.AddDistributedMemoryCache();
            services.AddSession(options =>
            {
                options.Cookie.Name = "NatterCookie";
                options.IdleTimeout = TimeSpan.FromMinutes(10);
                options.Cookie.HttpOnly = true;
                options.Cookie.IsEssential = false;
            });

            services.AddControllers(c =>
            {
                c.Filters.Add<EnforcePolicyAttribute>();
            });

            services.AddSwaggerGen(c => c.SwaggerDoc("v1", new OpenApiInfo { Title = "NatterApi", Version = "v1" }));

            services.AddHostedService<TokenJanitor>();

            services.AddOptions()
                .AddOptions<KeycloakOptions>()
                .BindConfiguration(KeycloakOptions.ConfigKey);

            services.AddRules();
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

            app.UseEndpoints(endpoints => endpoints.MapControllers());
        }
    }
}
