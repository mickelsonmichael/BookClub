using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using NatterApi.Services.TokenStore;

namespace NatterApi.Services
{
    /// <summary>
    /// 5.2.3 - Automatically deleting expired tokens using a scheduled task
    /// </summary>
    public class TokenJanitor : BackgroundService
    {
        public TokenJanitor(IServiceScopeFactory scopeFactory)
        {
            _scopeFactory = scopeFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromMinutes(10));

                using IServiceScope scope = _scopeFactory.CreateScope();

                ITokenService tokenService = scope.ServiceProvider.GetRequiredService<ITokenService>();

                await tokenService.ClearExpiredTokens();
            }
        }

        private readonly IServiceScopeFactory _scopeFactory;
    }
}
