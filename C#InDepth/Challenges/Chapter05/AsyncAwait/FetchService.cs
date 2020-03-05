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
            var keyword = new T().GetFetchKeyword();
            string fetchUrl = $"https://jsonplaceholder.typicode.com/{keyword}";

            using var httpClient = new HttpClient();
            string result = await httpClient.GetStringAsync(fetchUrl);
            
            return JsonConvert.DeserializeObject<List<T>>(result);
        }
    }
}