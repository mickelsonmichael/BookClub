using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace AsyncAwait
{
    public static class FetchService
    {
        public static async Task<List<T>> GetAll<T>() where T : IFetchable, new()
        {
            // create an instance so we can get the keyword from it
            var keyword = new T().GetFetchKeyword();
            string fetchUrl = $"https://jsonplaceholder.typicode.com/{keyword}";

            using var httpClient = new HttpClient(); // create the client
            string result = await httpClient.GetStringAsync(fetchUrl)
                                            .ConfigureAwait(false);
            
            // convert the JSON into the objects
            return JsonConvert.DeserializeObject<List<T>>(result);
        }
    }
}