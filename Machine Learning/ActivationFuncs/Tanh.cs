using System;

namespace ML.ActivationFunctions
{
    /// <summary>
    /// [-1, 1]
    /// </summary>
    public class AF_TanH : ActivationFunction
    {
        public override double Value(double sum)
            => Math.Tanh(sum);
        public override double Derivative(double sum)
        {
            double y = Value(sum);
            return 1.0 - y * y;
        }
    }
}
