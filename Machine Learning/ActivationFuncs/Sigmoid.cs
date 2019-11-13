using System;

namespace ML.ActivationFunctions
{
    /// <summary>
    /// [0, 1]
    /// </summary>
    public class AF_Sigmoid : ActivationFunction
    {
        public override double Value(double sum)
            => 1.0 / (1.0 + Math.Exp(-sum));
        public override double Derivative(double sum)
        {
            double output = Value(sum);
            return output * (1.0 - output);
        }
    }
}
