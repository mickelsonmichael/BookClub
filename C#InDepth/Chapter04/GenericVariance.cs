using System;

namespace Chapter04
{
    public static class GenericVariance
    {

        public static void Demo()
        {
            // there is an implicit conversion from object to string
            Func<object, int> original = (obj) => 1;
            Func<string, int> converted = original;

            // there is an identity conversion from dynamic to object
            // there is an implicit  conversion from string to IConvertible
            Func<dynamic, string> original2 = (dyn) => "hello";
            Func<object, IConvertible> converted2 = original2;

            // INVALLID
            // there is no implicit conversion from string to string
            //Func<string, int> original3 = (str) => 1;
            //Func<object, int> converted3 = original3;
        }

        public interface Covariant<out T>
        {
            public T Get();
        }

        public interface Contravariant<in T>
        {
            public void Add(T into);
        }

        public class Invariant<T> : Covariant<T>, Contravariant<T>
        {
            private T value;
            public void Add(T into) => value = into;
            public T Get() => value;
        }
    }
}