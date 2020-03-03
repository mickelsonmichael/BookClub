using System;
using System.Collections.Generic;
using System.Linq;

namespace Chapter04
{
    public static class Dynamics
    {
        public static void Demo()
        {
             dynamic text = "My text";
            var my = text.Substring(2);
            //var my = text.SUBSTR(2);

            // there is an implict conversion between dynamic and
            // any non-pointer type
            string str = text;

            // operations containing a dynamic result in a dynamic
            // in most cases and are bound at EXECUTION TIME
            var alsoDynamic = text + " is here";

            // even if it contains a dynamic parameter, 
            // constructors are bound at compile time
            var test = new TestObj(text);


            // in a static context, expando object is a dictionary
            IDictionary<string, object> expando = new System.Dynamic.ExpandoObject();
            expando.Add("testProperty", "myValue");
            Console.WriteLine(expando["testProperty"]);

            // in a dynamic context, expando object is dynamic
            dynamic dynamicExpando = expando;
            Console.WriteLine(dynamicExpando.testProperty);

            // can add keys dynamically even
            dynamicExpando.secondProperty = "thisToo";
            Console.WriteLine(expando["secondProperty"]);


            // anonymous methods can only be dynamic if casted
            // dynamic sqr = x => x * x;
            dynamic sqr = (Func<dynamic, dynamic>)(x => x * x);

            // same concept applies to LINQ methods
            // extension methods also cannot be called in a dynamic context
            List<dynamic> list = new List<dynamic> { 2, 4, 6 };
            // dynamic notHere = list.Select(x => x == "here");

            // to get around the lack of extension methods,
            // we can call them directly
            dynamic isHere = Enumerable.Any(list, x => sqr(x) > 30);
            Console.WriteLine(isHere);

            // expression trees cannot be used with dynamic
            // dynamic there = list.AsQueryable().First();

            Console.WriteLine("====");
        }

        public class TestObj
        {
            public TestObj(string str)
            {
            }
        }
    }
}