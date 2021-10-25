using System.ComponentModel.DataAnnotations;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using NatterApi.Exceptions;
using NatterApi.Models;
using NatterApi.Models.Requests;

namespace NatterApi.Controllers
{
    [ApiController, Route("/spaces/{spaceId:int}/messages")]
    public class MessageController : ControllerBase
    {
        public MessageController(NatterDbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [HttpPost]
        public IActionResult AddMessage(
            [FromBody, Required] CreateMessageRequest request,
            int spaceId
        )
        {
            // if (HttpContext.GetNatterUsername() != request.Author)
            // {
            //     return Unauthorized();
            // }

            Message message = request.ToMessage();

            Space space = GetSpace(spaceId);

            space.Messages!.Append(message);
            
            _dbContext.Update(space);
            _dbContext.SaveChanges();

            return Created($"/spaces/{spaceId}/messages/{message.MessageId}", message);
        }


        [HttpGet("/{messageId:int}")]
        public IActionResult GetMessage(int spaceId, int messageId)
        {
            Space space = GetSpace(spaceId);

            Message? message = space.Messages?.SingleOrDefault(m => m.MessageId == messageId);

            if (message == null)
            {
                throw new MessageNotFoundException(messageId);
            }

            return Ok(message);
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
    }
}