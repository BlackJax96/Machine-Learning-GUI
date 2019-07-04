using System;

namespace TheraEngine.Core.Maths.MachineLearning
{
    /// <summary>
    /// [-1, 1]
    /// </summary>
    public class AF_Sine : ActivationFunction
    {
        public override double Value(double sum)
        {
            return Math.Sin(sum);
        }
        public override double Derivative(double sum)
        {
            return Math.Cos(sum);
        }
    }
}
