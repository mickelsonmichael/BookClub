using System;

namespace NatterApi.Exceptions
{
    public class SpaceNotFoundException : NotFoundException
    {
        public SpaceNotFoundException(Guid spaceId)
            : base($"Unable to find space <{spaceId}>.")
        {
            // nothing doing
        }
    }
}
