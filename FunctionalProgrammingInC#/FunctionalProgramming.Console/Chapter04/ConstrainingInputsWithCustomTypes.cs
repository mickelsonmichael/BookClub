using System.ComponentModel.DataAnnotations;

using static System.Text.RegularExpressions.Regex;

namespace FunctionalProgramming.Console.Chapter04;

public record Age
{
    public int Value { get; }

    public Age(int value)
    {
        if (!IsValid(value))
        {
            // This is technicall a "dishonest" implementation because it does not specify that it can throw an exception
            throw new ArgumentException($"{value} is not a valid age.");
        }

        Value = value;
    }

    private static bool IsValid(int age) => 0 <= age && age <= 120;

    public static bool operator <(Age lhs, Age rhs) => lhs.Value < rhs.Value;
    public static bool operator >(Age lhs, Age rhs) => lhs.Value > rhs.Value;
    public static bool operator <(Age lhs, int rhs) => lhs < new Age(rhs);
    public static bool operator >(Age lhs, int rhs) => lhs > new Age(rhs);
}

public record AccountNumber
{
    public string Value { get; }
    public bool Valid { get; }

    public AccountNumber(string value)
    {
        // This is an "honest" function, but does not report any kind of feedback besides valid or invalid
        Valid = IsValid(value);
        Value = value;
    }

    private static bool IsValid(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return false;

        if (!IsMatch(value, "[0-9]{5}"))
            return false;

        return true;
    }
}

public record Birthday : IValidatableObject
{
    [Required]
    public DateTime Date { get; }

    // This is an honest implementation that also returns a collection of validation errors
    public Birthday(DateTime value)
    {
        // var errors = new List<string>();

        // if (value > DateTime.Now)
        //  errors.Add("Birthdays cannot be in the future!");

        // if (value == DateTime.MinValue)
        //  errors.Add("Birthdays cannot occur before history.");

        Date = value;
        // ValidationErrors = errors;
    }

    public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
    {
        var errors = new List<ValidationResult>();

        Validator.TryValidateProperty(this.Date, new ValidationContext(this, null, null)
        {
            MemberName = nameof(Date)
        }, errors);

        return errors;
    }
}

public record PhoneNumber
{
    public long Value { get; }

    // This is an honest implementation that utilizes a "null" object
    // https://refactoring.com/catalog/introduceSpecialCase.html
    protected PhoneNumber(long value)
    {
        Value = value;
    }

    public static PhoneNumber Parse(long n) => IsValid(n) ? new PhoneNumber(n) : new InvalidPhoneNumber();
    public static bool IsValid(long n) => IsMatch(n.ToString(), "^[0-9]{10}$");
}

public record InvalidPhoneNumber : PhoneNumber
{
    public InvalidPhoneNumber() : base(-1)
    {
    }
}
