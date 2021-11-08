using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NatterApi.Models.Token
{
    public class Session
    {
        public int SessionId { get; set; }
        public string? SessionCookieId { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime ExpireDate { get; set; }
        public string? UserName { get; set; }
    }
}
