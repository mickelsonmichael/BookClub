using Microsoft.AspNetCore.Mvc;

namespace Chapter05.NullVersesOption;

public class ExceptionPersonService
{
    public Person GetPerson(string name)
        => _people.FirstOrDefault(p => p.FirstName == name || p.LastName == name) ?? throw new Exception("not found");

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

        try
        {
            Person person = service.GetPerson(name);

            return Ok(person);
        }
        catch
        {
            return NotFound();
        }
    }
}
