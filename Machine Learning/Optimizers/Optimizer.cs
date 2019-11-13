namespace ML.Optimizers
{
    public abstract class Optimizer
    {
        public abstract double UpdateWeight(double weight, double prevWeight, double dTotRWt);
    }
}
