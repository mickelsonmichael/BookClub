using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace NatterApi.Models
{
    public class Group
    {
        [Key]
        public string GroupId { get; private set; }
        public ICollection<User> Users { get; private set; } = new List<User>();

        public Group(string groupId)
        {
            GroupId = groupId;
        }
    }
}
