using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using NatterApi.Extensions;
using NatterApi.Models;
using NRules;
using System;
using System.Collections.Generic;

namespace NatterApi.Filters
{
    /// <summary>
    /// 8.3 Attribute-based access control
    /// </summary>
    public class EnforcePolicyAttribute : IActionFilter
    {
        public EnforcePolicyAttribute(ISessionFactory sessionFactory)
        {
            _sessionFactory = sessionFactory;
        }

        public void OnActionExecuting(ActionExecutingContext context)
        {
            PopulateAttributes(context);

            Decision decision = CheckPermissions(context);

            if (!decision.IsPermitted)
            {
                context.Result = new UnauthorizedResult();
            }
        }

        private void PopulateAttributes(ActionExecutingContext context)
        {
            HttpContext httpContext = context.HttpContext;

            subjectAttributes["user"] = httpContext.GetNatterUsername();
            subjectAttributes["groups"] = httpContext.Items["groups"];

            resourceAttributes["path"] = httpContext.Request.Path;
            resourceAttributes["space"] = context.ModelState["spaceId"];

            actionAttributes["method"] = httpContext.Request.Method;

            environmentAttributes["timeOfDay"] = DateTime.Now;
            environmentAttributes["ip"] = httpContext.Connection.RemoteIpAddress;
        }

        public Decision CheckPermissions(ActionExecutingContext context)
        {
            NRules.ISession rulesSession = _sessionFactory.CreateSession();

            Decision decision = new();

            rulesSession.Insert(decision);

            rulesSession.InsertAll(new object[]
            {
                new Subject(subjectAttributes),
                new Resource(resourceAttributes),
                new ApiAction(actionAttributes),
                new ApiEnvironment(environmentAttributes)
            });

            rulesSession.Fire();

            return decision;
        }

        public void OnActionExecuted(ActionExecutedContext context)
        {
        }

        private readonly IDictionary<string, object?> subjectAttributes = new Dictionary<string, object?>();
        private readonly IDictionary<string, object?> actionAttributes = new Dictionary<string, object?>();
        private readonly IDictionary<string, object?> environmentAttributes = new Dictionary<string, object?>();
        private readonly IDictionary<string, object?> resourceAttributes = new Dictionary<string, object?>();
        private readonly ISessionFactory _sessionFactory;
    }
}
