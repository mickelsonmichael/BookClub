using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.ValueGeneration;
using Repo.Models;

namespace Repo.ValueGenerators
{
    public class FriendlyIdGenerator : ValueGenerator<string>
    {
        public override bool GeneratesTemporaryValues => false;

        public override string Next(EntityEntry entry)
        {
            var authorName = (string)entry.Property(nameof(Author.Name)).CurrentValue;
            IEnumerable<string> nameSegments = authorName.Split(' ');
            var lastName = nameSegments.Last();

            var uniqueId = $"{lastName.ToLower()}-{DateTime.Now.Ticks}";

            return uniqueId;
        }
    }
}