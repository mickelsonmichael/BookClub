
using System;
using Chapter04.Examples;

namespace Chapter04
{
    class Program
    {
        static void Main(string[] args)
        {
           Dynamics.Demo();

           Arguments.Demo();

           GenericVariance.Demo();

            Console.WriteLine("# Dynamics - JSON");

           string todo = DynamicJson.WhatToDo();

           Console.WriteLine(todo);

           DynamicJson.MakeAPost();

            Console.WriteLine("=====\n#Dynamics - Reflection");

           ReflectionDemo.Compare();
        }
    }

    
}
