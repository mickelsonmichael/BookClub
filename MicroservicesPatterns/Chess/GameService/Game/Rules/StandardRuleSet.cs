using GameService.Game.Pieces;
using GameService.Game.Positions;
using static GameService.Game.Players.Player;

namespace GameService.Game.Rules;

public class StandardRuleSet : IRuleSet
{
    public ICollection<IWinCondition> WinConditions => new[]
    {
        new CheckmateWinCondition()
    };

    public IEnumerable<IChessPiece> GetInitialPieces()
    {
        // | - || a | b | c | d | e | f | g | h |
        // | ---------------------------------- |
        // | 8 || R | N | B | K | Q | B | N | R |
        // | 7 || P | P | P | P | P | P | P | P |
        // | 6 || - | - | - | - | - | - | - | - |
        // | 5 || - | - | - | - | - | - | - | - |
        // | 4 || - | - | - | - | - | - | - | - |
        // | 3 || - | - | - | - | - | - | - | - |
        // | 2 || P | P | P | P | P | P | P | P |
        // | 1 || R | N | B | K | Q | B | N | R |

        yield return new Pawn(White, new Position('a', 2));
        yield return new Pawn(White, new Position('b', 2));
        yield return new Pawn(White, new Position('c', 2));
        yield return new Pawn(White, new Position('d', 2));
        yield return new Pawn(White, new Position('e', 2));
        yield return new Pawn(White, new Position('f', 2));
        yield return new Pawn(White, new Position('g', 2));
        yield return new Pawn(White, new Position('h', 2));

        yield return new Pawn(Black, new Position('a', 2));
        yield return new Pawn(Black, new Position('b', 2));
        yield return new Pawn(Black, new Position('c', 2));
        yield return new Pawn(Black, new Position('d', 2));
        yield return new Pawn(Black, new Position('e', 2));
        yield return new Pawn(Black, new Position('f', 2));
        yield return new Pawn(Black, new Position('g', 2));
        yield return new Pawn(Black, new Position('h', 2));
    }
}
