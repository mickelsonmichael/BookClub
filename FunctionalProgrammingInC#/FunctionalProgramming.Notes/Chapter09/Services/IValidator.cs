using LanguageExt;

namespace FunctionalProgramming.Notes.Chapter09.Services;

public interface IValidator<T>
{
    Validation<ICollection<string>, T> Validate(T input);
}
