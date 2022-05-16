using static System.Console;

// https://louthy.github.io/language-ext/LanguageExt.Core/Prelude/Currying%20and%20Partial%20Application/index.html
using static LanguageExt.Prelude;

// The using statements...allow us to attach some semantic meaning to specific uses of the string type...
// You could go the extra mile and define dedicated types...
// But for the present discussion, I’m not too worried about enforcing business rules...
using Name = System.String;
using Greeting = System.String;
using PersonalizedGreeting = System.String;

namespace FunctionalProgramming.Notes.Chapter09.Examples;

public class Greet
{
    // Field
    // Cannot be changed once class is initialized (can't depend on other instance variables)
    private readonly Func<Greeting, Name, PersonalizedGreeting> Greeter_Field
        = (greeting, name) => $"{greeting}, {name}";

    // Property
    // Can depend on other instance variables
    public Func<Greeting, Name, PersonalizedGreeting> Greeter_Prop
        => (greeting, name) => $"{greeting}, {name}";

    // Factory
    // Can depend on other instance variables
    Func<Greeting, T, PersonalizedGreeting> GreeterFactory<T>()
        => (greeting, t) => $"{greeting}, {t}";

    public void Do()
    {
        // ======================================================
        // 9.1 Partial application: Supplying arguments piecemeal
        // ======================================================
        Name[] names = { "Mike", "Lyn" };

        // (Greeting, Name) -> PersonalizedGreeting
        var greet = (Greeting gr, Name name) => $"{gr}, {name}";

        names.Map(name => greet("Hello", name))
            .Iter(WriteLine); // NOTE: LanguageExt uses Iter instead of ForEach

        // ===========================================
        // 9.1.1 Manually enabling partial application
        // ===========================================
        {
            // Greeting -> (Name -> PersonalizedGreeting)
            //
            // OR (arrow notation is right-associative)
            //
            // Greeting -> Name -> PersonalizedGreeting
            Func<Name, PersonalizedGreeting> greetWith(Greeting gr) => (Name name) => $"{gr}, {name}";

            Func<Name, PersonalizedGreeting> greetFormally = greetWith("Salutations");

            names.Map(name => greetFormally(name))
                .Iter(WriteLine);

            // <definition>
            // Curried
            // - All arguments are supplied one at a time via function invocation
            // </definition>
        }

        // ======================================
        // 9.1.2 Generalizing partial application
        // ======================================
        {
            var greetInformally = par(greet, "Hey"); // NOTE: LanguageExt provides par for partial application

            names.Map(greetInformally).Iter(WriteLine);
        }

        // ==============================================
        // 9.2 Overcoming the quirks of method resolution
        // ==============================================
        {
            PersonalizedGreeting GreeterMethod(Greeting gr, Name name)
                => $"{gr}, {name}";

            // DOES NOT COMPILE
            // Apply() expects a Func not a local method
            //Func<Name, PersonalizedGreeting> GreetWith(Greeting greeting)
            //    => GreeterMethod.Apply(greeting);

            // DOES NOT COMPILE
            // Func<Name, PersonalizedGreeting> GreetWith_1(Greeting gr)
            //     => ListExtensions.Apply<Greeting,Name,PersonalizedGreeting>(GreeterMethod, gr);

            Func<Name, PersonalizedGreeting> GreetWith_2(Greeting gr)
                => par(new Func<Greeting, Name, PersonalizedGreeting>(GreeterMethod), gr);

            var _ = GreetWith_2("nope");

            // Field
            var greetWithField = par(Greeter_Field, "Field");
            // Property
            var greetWithProp = par(Greeter_Prop, "Property");
            // Factor
            var greetWithFactory = par(GreeterFactory<Name>(), "Hi");
        }
    }
}
