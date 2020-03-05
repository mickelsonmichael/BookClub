
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace AsyncAwait
{
    public static class TodoService
    {
        private static List<todo> Cache;

        public static async Task<List<todo>> GetAll()
        {
            if (Cache != null) return Cache;

            const string fetchUrl = "https://jsonplaceholder.typicode.com/todos";

            using var httpClient = new HttpClient();
            string result = await httpClient.GetStringAsync(fetchUrl);
            
            Cache = JsonConvert.DeserializeObject<List<todo>>(result);

            return Cache;
        }
    }
}
