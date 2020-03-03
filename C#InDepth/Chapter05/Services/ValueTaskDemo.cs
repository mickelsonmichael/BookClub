using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Chapter05.Services
{
    /// A method may return an instance of ValueType when it's likely 
    // that the result of its operation will be available synchronously, 
    // and when it's expected to be invoked so frequently 
    // that the cost of allocating a new Task<TResult> for each call will 
    // be prohibitive.
    public static class ValueTaskDemo
    {
        public static async Task Demo()
        {
            var numbers = new List<int>();
            var towait = TimeSpan.FromMilliseconds(10);

            for (int i = 0; i < 1_000; i++)
            {
                var number = GetNumber(); // start the task

                // ideally there would be some logic here taking time

                // if number is available, then we don't have to allocate a Task
                // which leads to faster code with less memory usage
                // however, if the calculation isn't done yet,
                // then we will have to allocate one and await it
                numbers.Add(await number);
            }

            Console.WriteLine($"Retrieved {numbers.Count} numbers!");
        }

        private static ValueTask<int> GetNumber()
        {
            // this could be any method that returns quickly
            // here it returns instantly with 0 calculations
            // this could also be done with a cache
            return new ValueTask<int>(101);
        }
    }
}