namespace NatterApi.Exceptions
{
    public class MessageNotFoundException : NotFoundException
    {
        public MessageNotFoundException(int messageId)
            : base($"Unable to find message <{messageId}>.")
        {
            // nothing doing
        }
    }
}
