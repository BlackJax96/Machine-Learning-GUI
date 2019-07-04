namespace TheraEngine.Core.Maths.MachineLearning
{
    //https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
    public abstract class ActivationFunction
    {
        public abstract double Value(double sum);
        public abstract double Derivative(double sum);
    }
}
