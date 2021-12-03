using System.ComponentModel.DataAnnotations.Schema;

namespace NatterApi.Models
{
    public class GroupMember
    {
        public string GroupId { get; private set; }
        public string Username { get; private set; }
        [ForeignKey(nameof(GroupId))]
        public Group? Group { get; private set; }

        public GroupMember(string groupId, string username)
        {
            GroupId = groupId;
            Username = username;
        }
    }
}
