namespace NatterApi.Exceptions
{
    public class SpaceNotFoundException : NotFoundException
    {
        public SpaceNotFoundException(int spaceId)
            : base($"Unable to find space <{spaceId}>.")
        {
            // nothing doing
        }
    }
}
