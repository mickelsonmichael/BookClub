namespace Chapter03
{
    public delegate double ReadDouble();

    public class Pure
    {
        public double Height;
        public double Weight;
        public double BMI;
        public string Health;

        public Pure(
            ReadDouble height,
            ReadDouble weight
        )
        {
            Height = height();
            Weight = weight();
            BMI = Weight / (Height * Height);

            Health = BMI switch
            {
                < 18.5 => "underweight",
                >= 25 => "overweight",
                _ => "healthy",
            };
        }
    }
}
