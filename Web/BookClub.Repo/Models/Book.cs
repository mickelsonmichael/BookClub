using System.ComponentModel.DataAnnotations;

namespace Repo.Models
{
    public class Book
    {
        public string BookId { get; set; }
        [Required]
        public string Name { get; set; }
        public string Edition { get; set; }
        [Required]
        public string Author { get; set; }
        public string Image { get; set; }
        public bool Current { get; set; }
        public bool Complete { get; set; }
    }
}
