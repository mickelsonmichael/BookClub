using Microsoft.Extensions.DependencyInjection;
using NatterApi.Models.Rules;
using NRules;
using NRules.Fluent;
using System.Reflection;

namespace NatterApi.Extensions
{
    public static class AddRulesExtension
    {
        /// <summary>
        /// 8.3.2 Implementing ABAC decisions
        /// </summary>
        public static IServiceCollection AddRules(this IServiceCollection services)
        {
            return services.AddSingleton(services =>
            {
                RuleRepository ruleRepository = new();

                ruleRepository.Load(x => x.From(Assembly.GetExecutingAssembly()));

                return ruleRepository.Compile();
            });
        }
    }
}
