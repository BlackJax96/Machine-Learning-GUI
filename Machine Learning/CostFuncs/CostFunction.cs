namespace ML.CostFunctions
{
    public abstract class CostFunction
    {
        public abstract double Evaluate(double target, double actual);
        public abstract double Derivative(double target, double actual);
    }
}
