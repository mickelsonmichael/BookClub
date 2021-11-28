using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Mvc;
using NatterApi.Exceptions;
using NatterApi.Extensions;
using NatterApi.Filters;
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

        [HttpGet]
        [AuthFilter]
        public IActionResult GetSpaces()
        {
            List<Space> spaces = _context.Spaces.ToList();

            return Ok(spaces);
        }

        // Page 36
        [HttpPost]
        [RequireScope("create_space")]
        [AuthFilter]
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

            Permission permission = new(space.Id, username, "rwd");

            _context.Add(permission);
            _context.SaveChanges();

            string url = $"/spaces/{space.Id}";

            return Created(
                url,
                new { name = space.Name, uri = url }
            );
        }

        [HttpGet("{spaceId:int}")]
        [AuthFilter(AccessLevel.Read)]
        public IActionResult GetSpace(int spaceId)
        {
            Space? space = _context.Spaces.Find(spaceId);

            if (space == null)
            {
                throw new SpaceNotFoundException(spaceId);
            }

            return Ok(space);
        }

        [HttpPost("{spaceId:int}/members")]
        [RequireScope("add_member")]
        [AuthFilter(AccessLevel.All)]
        public IActionResult AddMember(
            [FromBody, Required] AddMemberRequest request,
            int spaceId
        )
        {
            if (!Regex.IsMatch(request.Permissions, "r?w?d?"))
            {
                return BadRequest($"Invalid permissions \"{request.Permissions}\"");
            }

            User? user = _context.Users.Find(request.Username);

            if (user == null)
            {
                return BadRequest("Invalid username.");
            }

            Space? space = _context.Spaces.Find(spaceId);

            if (space == null)
            {
                return NotFound($"Could not find space with ID {spaceId}");
            }

            Permission? currentPermission = _context.Permissions.Find(spaceId, request.Username);

            if (currentPermission != null)
            {
                _context.Remove(currentPermission);
            }

            Permission permission = new(spaceId, request.Username, request.Permissions);
            _context.Add(permission);

            _context.SaveChanges();

            return Ok();
        }

        private readonly NatterDbContext _context;
    }
}