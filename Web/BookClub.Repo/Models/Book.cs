using System;
using System.ComponentModel.DataAnnotations;

namespace Repo.Models
{
    public class Book
    {
        public string BookId { get; set; }
        [Required]
        public string Name { get; set; }
        [ConcurrencyCheck]
        public string Edition { get; set; }
        public string Image { get; set; }
        public bool Current { get; private set; }
        public DateTime? StartDate { get; set; }
        public DateTime? CompletedDate { get; set; }
        public DateTime CreatedDate { get; private set; }
        [Required]
        public Author Author { get; set; }
    }
}
