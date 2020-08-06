using System;

namespace Chapter15
{
    public class SwitchExpressions
    {
        public string AsExpressionBody(string test)
             => test switch
             {
                 "hello" => "Hi",
                 "how are you" => "Good, you?",
                 _ => throw new NotSupportedException("Not supported input")
             };

        public void AsInlineExpression()
        {
            string test = "testing";

            var result = test switch
            {
                "not a test" => 1000,
                "testing" => 2000,
                "done" => 3000,
                _ => 0
            };

            Console.WriteLine(result);
        }

        public string AsPattern(Shape s)
            => s switch
            {
                Triangle t => "triangle",
                Square sq => "square",
                _ => "I don't know?"
            };
    }

    public class Shape
    {

    }

    public class Triangle : Shape
    {

    }

    public class Square : Shape
    {

    }
}