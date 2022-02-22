using Microsoft.AspNetCore.Mvc;

namespace Chapter05.NullVersesOption;

public class NullPersonService
{
    public Person? GetPerson(string name)
        => _people.FirstOrDefault(p => p.FirstName == name || p.LastName == name);

    public void AddPerson(Person p)
        => _people.Add(p);

    private readonly ICollection<Person> _people = new List<Person>();
}

public class NullController : Controller
{
    [HttpGet("/{name:string}")]
    public IActionResult GetPerson(string name)
    {
        var service = new NullPersonService();

        Person? person = service.GetPerson(name);

        return person == null
            ? NotFound()
            : Ok();
    }
}
