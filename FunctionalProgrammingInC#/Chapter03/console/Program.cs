using Chapter03;
using static System.Console;

// 1. Prompt the user for their height in meters and weight in kilograms
Write("Enter your height in meters: ");
double height = double.Parse(ReadLine() ?? "2");

Write("Enter your weight in kilograms: ");
double weight = double.Parse(ReadLine() ?? "70");

// 2. Calculate the BMI as weight / height^2
double bmi = weight / (height * height);

// 3. Output a message: underweight (BMI < 18.5), overweight (BMI >= 25), or healthy
switch (bmi)
{
    case < 18.5:
        WriteLine("underweight");
        break;
    case >= 25:
        WriteLine("overweight");
        break;
    default:
        WriteLine("healthy");
        break;
}

// 4. Structure your code that pure and impore parts are separate
Pure body = new(
    () => {
        Write("Enter your height in meters: ");
        while (true)
        {
            if (double.TryParse(ReadLine(), out double result))
            {
                return result;
            }
        }
    },
    () => {
        Write("Enter your weight in kilograms: ");
        return double.Parse(ReadLine() ?? "1");
    });

WriteLine(body.Health);

// 5. Unit test the pure parts
// See PureTests.cs