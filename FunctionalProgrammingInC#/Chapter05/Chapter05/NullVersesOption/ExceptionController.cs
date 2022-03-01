using Microsoft.AspNetCore.Mvc;

namespace Chapter05.NullVersesOption;

public class NotFoundException : Exception
{
    public NotFoundException(string message) : base(message)
    { }
}

public class ExceptionPersonService
{
    ///<exception cref="NotFoundException" />
    public Person GetPerson(string name)
        => _people.FirstOrDefault(p => p.FirstName == name || p.LastName == name)
            ?? throw new NotFoundException("not found");

    public void AddPerson(Person p)
        => _people.Add(p);

    private readonly ICollection<Person> _people = new List<Person>();
}

public class ExceptionController : Controller
{
    [HttpGet("/{name:string}")]
    public IActionResult GetPerson(string name)
    {
        var service = new ExceptionPersonService();

        Person person;
        try
        {
            person = service.GetPerson(name);
        }
        catch (NotFoundException)
        {
            return NotFound();
        }

        string _ = person.FirstName; // do stuff with person

        return Ok(person);
    }
}
