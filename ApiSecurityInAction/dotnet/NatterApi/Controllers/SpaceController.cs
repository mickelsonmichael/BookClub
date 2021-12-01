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

            // 3.2.2 Static Roles
            UserRole role = new(space.Id, "owner", username);
            _context.Add(role);
            
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
            RolePermission? role = _context.RolePermissions.Find(request.Role);

            if (role == null)
            {
                return BadRequest($"Invalid role \"{request.Role}\"");
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

            UserRole? currentRole = _context.UserRoles.Find(spaceId, request.Username);

            if (currentRole != null)
            {
                _context.Remove(currentRole);
            }

            UserRole newRole = new(spaceId, role.RoleId, request.Username);

            _context.UserRoles.Add(newRole);

            _context.SaveChanges();

            return Ok();
        }

        private readonly NatterDbContext _context;
    }
}