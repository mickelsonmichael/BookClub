using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using NatterApi.Exceptions;
using NatterApi.Extensions;
using NatterApi.Filters;
using NatterApi.Models;
using NatterApi.Models.Requests;

namespace NatterApi.Controllers
{
    [ApiController, Route("/spaces/{spaceId:int}/messages")]
    [ValidateAntiForgeryToken]
    public class MessageController : ControllerBase
    {
        public MessageController(NatterDbContext dbContext, ILogger<MessageController> logger)
        {
            _dbContext = dbContext;
            _logger = logger;
        }

        [HttpGet]
        public IActionResult GetMessages(
            int spaceId,
            [FromQuery] DateTime? since
        )
        {
            _logger.LogInformation("Request for messages from space {SpaceId} since {Since}.", spaceId, since?.ToString() ?? "first message");

            Space space = GetSpace(spaceId);

            List<Message> messages = space.Messages
                ?.Where(m => m.MessageTime >= since).ToList()
                ?? new List<Message>();

            return Ok(messages);
        }

        [HttpPost]
        public IActionResult AddMessage(
            [FromBody, Required] CreateMessageRequest request,
            int spaceId
        )
        {
            if (HttpContext.GetNatterUsername() != request.Author)
            {
                return Unauthorized();
            }

            _logger.LogInformation("Requested addition of message to {SpaceId}.\n{Message}", spaceId, request);

            Message message = request.ToMessage();

            Space space = GetSpace(spaceId);

            space.Messages!.Add(message);

            _dbContext.Update(space);
            _dbContext.SaveChanges();

            return Created($"/spaces/{spaceId}/messages/{message.MessageId}", message);
        }

        [HttpGet("{messageId:int}"), AuthFilter(AccessLevel.Read)]
        public IActionResult GetMessage(int spaceId, int messageId)
        {
            _logger.LogInformation("Requested details for space {SpaceId} message {MessageId}.", spaceId, messageId);

            Space space = GetSpace(spaceId);

            Message? message = space.Messages?.SingleOrDefault(m => m.MessageId == messageId);

            if (message == null)
            {
                throw new MessageNotFoundException(messageId);
            }

            return Ok(message);
        }

        [HttpDelete("{messageId:int}")]
        public IActionResult DeleteMesage(int spaceId, int messageId)
        {
            _logger.LogInformation("Requested deletion of message {MessageId} from {SpaceId}.", messageId, spaceId);

            Space space = GetSpace(spaceId);

            Message? message = space.Messages?.SingleOrDefault(m => m.MessageId == messageId);

            if (message == null)
            {
                throw new MessageNotFoundException(messageId);
            }

            space.Messages!.Remove(message);

            _dbContext.Update(space);
            _dbContext.SaveChanges();

            return NoContent();
        }

        private Space GetSpace(int spaceId)
        {
            Space? space = _dbContext.Spaces.Include(s => s.Messages)
                .SingleOrDefault(s => s.Id == spaceId);

            if (space == null)
            {
                throw new SpaceNotFoundException(spaceId);
            }

            return space;
        }

        private readonly NatterDbContext _dbContext;
        private readonly ILogger<MessageController> _logger;
    }
}