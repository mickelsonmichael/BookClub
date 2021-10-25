using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Exceptions;
using NatterApi.Extensions;
using NatterApi.Filters;
using NatterApi.Models;
using NatterApi.Models.Requests;

namespace NatterApi.Controllers
{
    [ApiController, Route("/spaces"), AuthFilter]
    public class SpaceController : ControllerBase
    {
        public SpaceController(NatterDbContext context)
        {
            _context = context;
        }

        [HttpGet]
        public IActionResult GetSpaces()
        {
            List<Space> spaces = _context.Spaces.ToList();

            return Ok(spaces);
        }

        // Page 36
        [HttpPost]
        public IActionResult CreateSpace(
            [FromBody, Required] CreateSpaceRequest request
        )
        {
            string? username = HttpContext.GetNatterUsername();

            if (username == null || request.Owner != username)
            {
                return Unauthorized();
            }

            Space space = request.CreateSpace();

            _context.Add(space);
            _context.SaveChanges();

            string url = $"/spaces/{space.Id}";

            return Created(
                url,
                new { name = space.Name, uri = url }
            );
        }

        [HttpGet("/{spaceId:int}")]
        public IActionResult GetSpace(int spaceId)
        {
            Space? space = _context.Spaces.Find(spaceId);

            if (space == null)
            {
                throw new SpaceNotFoundException(spaceId);
            }

            return Ok(space);
        }

        private readonly NatterDbContext _context;
    }
}