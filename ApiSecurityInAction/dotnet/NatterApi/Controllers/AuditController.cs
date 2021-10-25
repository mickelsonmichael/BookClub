using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Models;

namespace NatterApi.Controllers
{
    [ApiController, Route("/logs")]
    public class AuditController : ControllerBase
    {
        public AuditController(NatterDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [HttpGet]
        public IActionResult GetLogs()
        {
            DateTime since = DateTime.Now.AddHours(-1);

            List<AuditMessage> logs = _dbContext.AuditLog
                .Where(log => log.AuditTime >= since)
                .Take(20)
                .ToList();

            return Ok(logs);
        }

        private readonly NatterDbContext _dbContext;
    }
}
