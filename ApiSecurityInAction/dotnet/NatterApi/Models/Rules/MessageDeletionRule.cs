using NRules.Fluent.Dsl;
using System;

namespace NatterApi.Models.Rules
{
    /// <summary>
    /// 8.3.2 Implementing ABAC decisions
    /// </summary>
    public class MessageDeletionRule : Rule
    {
        public override void Define()
        {
            ApiAction action = new();
            ApiEnvironment env = new();
            Decision decision = new();

            When()
                .Match<Decision>(() => decision)
                .Match<ApiAction>(
                    () => action,
                    act => act["method"] as string == "DELETE"
                )
                .Match<ApiEnvironment>(
                    () => env,
                    env => (env["timeOfDay"] as DateTime?).HasValue 
                        && (
                            ((DateTime)env["timeOfDay"]!).Hour > 17
                            || ((DateTime)env["timeOfDay"]!).Hour < 9
                        )
                );

            Then()
                .Do(_=> decision.Deny());
        }
    }
}
