using Newtonsoft.Json;
using System.Net;

namespace Chapter04
{
    public static class CommentService
    {
        public static object GetComments()
        {
            const string fetchUrl = "https://jsonplaceholder.typicode.com/comments";
            
            using var client = new WebClient();

            var str = client.DownloadString(fetchUrl);

            return JsonConvert.DeserializeObject(str);
        }
    }
}