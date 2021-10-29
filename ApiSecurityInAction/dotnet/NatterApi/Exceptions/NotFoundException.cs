using System;

namespace NatterApi.Exceptions
{
    public class NotFoundException : Exception
    {
        public NotFoundException(string message) : base(message)
        {
            // nothing doing
        }
    }
}