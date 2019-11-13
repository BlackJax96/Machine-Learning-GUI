using System;

namespace ML.ActivationFunctions
{
    /// <summary>
    /// [-1, 1]
    /// </summary>
    public class AF_Logistic : ActivationFunction
    {
        public override double Value(double sum)
            => 1.0 / (1.0 + Math.Exp(-sum));
        public override double Derivative(double sum)
        {
            double value = Value(sum);
            return value * (1.0 - value);
        }
    }
}
