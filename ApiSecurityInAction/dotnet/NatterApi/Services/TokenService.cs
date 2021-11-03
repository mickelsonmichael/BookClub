using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NatterApi.Models;

namespace NatterApi.Services
{
    public class TokenService
    {
        public TokenService()
        {

        }

        public Token CreateToken()
        {
            return new Token();
        }

        public bool ReadToken()
        {
            return true;
        }

        public void DeleteToken()
        {

        }
    }
}
