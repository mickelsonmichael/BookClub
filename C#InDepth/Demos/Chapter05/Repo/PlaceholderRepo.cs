using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Chapter05.Business;

namespace Chapter05.Repo
{
    public class PlaceholderRepo : IPlaceholderRepo
    {
        HttpClient Client = new HttpClient();
        const string FetchUrl = "https://jsonplaceholder.typicode.com/comments";

        public async Task<IEnumerable<Comment>> GetCommentsAsync()
        {
            Console.WriteLine("Requesting Items");
            var stringTask = Client.GetStringAsync(FetchUrl)
                .ConfigureAwait(false);
        
            Console.WriteLine("Deserializing request...");
            return JsonSerializer
                .Deserialize<List<Comment>>(await stringTask);
        }
    }
}