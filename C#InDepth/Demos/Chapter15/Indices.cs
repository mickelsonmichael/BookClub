using System.Linq;

namespace Chapter15
{
    public class Indices
    {
        public void LookAtThis()
        {
            var list = Enumerable.Range(1, 100).ToArray();

            var first10 = list[1..10];
            var last3 = list[^3..];
            var lastOne = list[^1];
            var first20 = list[..20];
            var skip1 = list[1..];
        }
    }
}