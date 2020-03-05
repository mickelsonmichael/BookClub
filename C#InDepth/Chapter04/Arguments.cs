using System;

namespace Chapter04
{
    public static class Arguments
    {
        /// i've used local functions here to keep the calls as close to the
        /// signatures as possible. If you aren't familiar with local methods that's
        /// OK, they're a C# 7 feature that isn't often used.! Just know that the
        /// `void WithDefault(...)` line is not a variable, it is a method
        public static void Demo()
        {
            Console.WriteLine("# Arguments");

            // you can use default(type) to define an optional value
            WithDefault();
            void WithDefault(int val = default(int)) => Console.WriteLine(val); // OUTPUT "0"

            // in c# 7+ you can simply use the default keyword
            WithDefault7();
            void WithDefault7(int val = default) => Console.WriteLine(val); // OUTPUT "0"

            // you may also use a constructor for value types
            WithConstructor();
            void WithConstructor(Guid val = new Guid()) => Console.WriteLine(val.ToString()); // OUTPUT "00000000-0000-0000-0000-000000000000"

            // you can specify named arguments in any order
            AnyOrder(second: 2, first: 1); // OUTPUT "1 2"

            // arguments are evalutated one at a time, and can be modified before the next one is evaluated
            var i = 0;
            AnyOrder(i++, i); // OUTPUT "0 1"

            void AnyOrder(int first, int second) => Console.WriteLine($"{first} {second}");

            // All of the following are invalid:
            // void InvalidRef(ref object obj) { Invalid }
            // void InvalidOut(out object obj) { Invalid }
            // void InvalidConstructor(Person p = new Person());

            Console.WriteLine("=====");
        }
    }
}
