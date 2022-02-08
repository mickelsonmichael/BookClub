using Xunit;

namespace Chapter03.Tests
{
    public class PureTests
    {
        [Fact]
        public void Pure_BmiCorrect()
        {
            const double h = 10;
            const double w = 70;
            const double bmi = w / (h * h);

            Pure p = new(
                () => h,
                () => w
            );

            Assert.Equal(bmi, p.BMI);
        }

        [Theory]
        [InlineData(50, "underweight")]
        [InlineData(90, "healthy")]
        [InlineData(120, "overweight")]
        public void Pure_HealthCorrect(double weight, string health)
        {
            const double h = 2;

            Pure p = new(
                () => h,
                () => weight
            );

            Assert.Equal(health, p.Health);
        }
    }
}
