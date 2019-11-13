using System;

namespace ML.ActivationFunctions
{
    /// <summary>
    /// [-1, 1]
    /// </summary>
    public class AF_Sine : ActivationFunction
    {
        public override double Value(double sum)
            => Math.Sin(sum);
        public override double Derivative(double sum)
            => Math.Cos(sum);
    }
}
