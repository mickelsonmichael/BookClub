namespace AsyncAwait
{
    public class user : IFetchable
    {
        public int id { get; set; }
        public string username { get; set; }
        public string name { get; set; }
        public string email { get; set; }

        public string GetFetchKeyword() => "users";
    }
}
