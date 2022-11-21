namespace GameService.Game.Positions;

public record Position
{
    public bool IsValid => (Number is > 0 and < 9) && (Letter is >= 'a' and <= 'h');

    public char Letter { get; private init; }

    public int Number { get; private init; }

    public Position(char letter, int number)
    {
        Letter = char.ToLower(letter);
        Number = number;
    }

    public Position Down() => this with { Number = Number - 1 };

    public Position Up() => this with { Number = Number + 1 };

    public Position Left() => this with { Letter = (char)(Letter - 1) };

    public Position Right() => this with { Letter = (char)(Letter + 1) };

    public string Stringify() => $"{char.ToUpper(Letter)}{Number}";

    public static Position Invalid() => new('z', 0);

    public static Position Parse(string positionString)
    {
        if (positionString.Length != 2)
        {
            return Invalid();
        }

        char[] coords = positionString.ToCharArray();

        if (!int.TryParse(coords[1].ToString(), out int n))
        {
            return Invalid();
        }

        return new(coords[0], n);
    }
}
