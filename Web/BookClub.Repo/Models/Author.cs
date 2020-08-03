using System.Collections.Generic;

namespace Repo.Models
{
    public class Author
    {
        public string AuthorId { get; set; }
        public string FriendlyId { get; private set; }
        public string Name { get; set; }
        public ICollection<Book> Books { get; set; }
    }
}