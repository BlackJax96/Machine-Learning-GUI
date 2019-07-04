namespace TheraEngine.Core.Maths.MachineLearning
{
    /// <summary>
    /// [-infinity, infinity)
    /// </summary>
    public class AF_LeakyReLU : ActivationFunction
    {
        public double Slope { get; set; } = 0.2;

        public override double Value(double sum)
        {
            if (sum >= 0.0)
                return sum;
            else
                return sum * Slope;
        }
        public override double Derivative(double sum)
        {
            if (sum >= 0.0)
                return 1.0;
            else
                return Slope;
        }
    }
}
