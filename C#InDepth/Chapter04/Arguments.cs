using System;

namespace Chapter04
{
    public static class Arguments
    {
        public static void Demo()
        {
            // you can use default(type) to define an optional value
            WithDefault();

            // in c# 7+ you can simply use the default keyword
            WithDefault7();

            // you may also use a constructor for value types
            WithConstructor();

            AnyOrder(second: 2, first: 1);

            // arguments are evalutated one at a time
            var i = 0;
            AnyOrder(i++, i);
        }

        // public void InvalidRef(ref object obj) { Invalid }
        // public void InvalidOut(out object obj) { Invalid }
        public static void WithDefault(int val = default(int)) => Console.WriteLine(val);
        public static void WithDefault7(int val = default) => Console.WriteLine(val);
        public static void WithConstructor(Guid val = new Guid()) => Console.WriteLine(val.ToString());
        // public static void InvalidConstructor(Person p = new Person());
        public static void AnyOrder(int first, int second) => Console.WriteLine($"{first} {second}");
    }
}