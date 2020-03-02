using Newtonsoft.Json;
using System.Collections.Generic;
using Sample.Business;
using System.IO;
using System.Threading.Tasks;

namespace Sample.Repo
{
    public class FileReader : IFileReader
    {
        public async Task<IEnumerable<Book>> GetBooksAsync()
        {
            var text = await File.ReadAllTextAsync("SampleData.json");

            return JsonConvert
                .DeserializeObject<BookSet>(text)
                .Books;
        }
    }
}