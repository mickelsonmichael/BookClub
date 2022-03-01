using Microsoft.AspNetCore.Mvc;

namespace Chapter05.NullVersesOption;

using static Chapter05.Option<Person>;

public class OptionPersonService
{
    public Option<Person> GetPerson(string name)
        => (Option<Person>)_people.FirstOrDefault(p => p.FirstName == name || p.LastName == name) ?? None;

    public void AddPerson(Person p)
        => _people.Add(p);

    private readonly ICollection<Person> _people = new List<Person>();
}

public class OptionController : Controller
{
    [HttpGet("/{name:string}")]
    public IActionResult GetPerson(string name)
    {
        var service = new OptionPersonService();

        Option<Person> person = service.GetPerson(name);

        return person == null
            ? NotFound()
            : Ok();
    }
}
