namespace TheraEngine.Core.Maths.MachineLearning
{
    /// <summary>
    /// [0, infinity)
    /// </summary>
    public class AF_ReLU : ActivationFunction
    {
        public override double Value(double sum)
        {
            if (sum < 0.0)
                return 0.0;
            return sum;
        }
        public override double Derivative(double sum)
        {
            if (sum < 0.0)
                return 0.0;
            return 1.0;
        }
    }
}
