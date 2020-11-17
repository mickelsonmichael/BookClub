using System;


namespace Chapter15
{
    public class DefaultInterfaceMethods : IImplementMe
    {
        public void DoStuff()
        {
            Console.WriteLine("Doing stuff");
        }
    }

    public interface IImplementMe
    {
        void DoStuff();
    }
}