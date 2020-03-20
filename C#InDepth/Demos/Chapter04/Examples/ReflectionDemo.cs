using System;
using System.Reflection;

namespace Chapter04.Examples
{
    public static class ReflectionDemo
    {
        private static object toReflect = new ObjectToReflect();

        public static void Compare()
        {
            WithoutDynamic();
            WithDynamic();
        }

        public static void WithoutDynamic()
        {
            // have to get the type
            Type type = toReflect.GetType();

            // find the method you want to call on the type
            MethodInfo method = type.GetMethod("WriteToConsole");

            // create a list of arguments to pass to the method
            object[] args = new object[] { "Without Test" };

            // invoke the method
            method.Invoke(toReflect, args);

            // get a property value
            PropertyInfo prop = type.GetProperty("MyProperty");
            var propValue = prop.GetValue(toReflect);
            Console.WriteLine(propValue);

            // update a property value
            PropertyInfo prop2 = type.GetProperty("MyProperty");
            prop.SetValue(toReflect, "My value has been updated with reflection without dynamics");

            // bring it all together
            // i could have reused the resources from earlier,
            // but I wanted to demonstrate how it would look as a
            // one-line statement. Spoiler: it's terrible
            toReflect.GetType()
                .GetMethod("WriteToConsole")
                .Invoke(toReflect, new object[] 
                    { 
                        toReflect.GetType().GetProperty("MyProperty").GetValue(toReflect) 
                    });
        }

        public static void WithDynamic()
        {
            // cast the object to a dynamic
            dynamic obj = toReflect;

            // invoke a method
            obj.WriteToConsole("With Test");

            // get a property value
            Console.WriteLine(obj.MyProperty);

            // update a preopty value
            obj.MyProperty = "I've been changed with dynamics";

            // bring it all together
            obj.WriteToConsole(obj.MyProperty);
        }

        private class ObjectToReflect
        {
            public string MyProperty { get; set; } = "This is my property";

            public void WriteToConsole(string stringToWrite)
            {
                Console.WriteLine(stringToWrite);
            }
        }
    }
}