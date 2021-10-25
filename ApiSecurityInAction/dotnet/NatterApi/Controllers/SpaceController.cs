using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Exceptions;
using NatterApi.Models;
using NatterApi.Models.Requests;

namespace NatterApi.Controllers
{
    [ApiController, Route("/spaces")]
    public class SpaceController : ControllerBase
    {
        public SpaceController(NatterDbContext context)
        {
            _context = context;
        }

        // Page 36
        [HttpPost]
        public IActionResult CreateSpace(
            [FromBody, Required] CreateSpaceRequest request
        )
        {
            Space space = request.CreateSpace();

            _context.Add(space);

            string url = $"/spaces/{space.Id}";

            return Created(
                url,
                new { name = space.Name, uri = url }
            );
        }

        [HttpGet("/{spaceId:guid}")]
        public IActionResult GetSpace(
            Guid spaceId // by using a Guid instead of a string, we let ASP.NET handle dangerous values for us automatically
        )
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