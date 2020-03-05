namespace AsyncAwait
{
    public class todo : IFetchable
    {
        public int userId { get; set; }
        public int id { get; set; }
        public string title { get; set; }
        public bool completed { get; set; }

        public string GetFetchKeyword() => "todos";
    }
}
