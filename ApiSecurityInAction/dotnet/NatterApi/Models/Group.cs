using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models
{
    public class Group
    {
        [Key]
        public string GroupId { get; private set; }
        public ICollection<Group> Groups { get; private set; } = new List<Group>();

        public Group(string groupId)
        {
            GroupId = groupId;
        }
    }
}
