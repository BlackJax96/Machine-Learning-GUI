namespace ML.Optimizers
{
    public class O_StaticLearningRate : Optimizer
    {
        public double LearningRate { get; set; }

        public override double UpdateWeight(double weight, double prevWeight, double dTotRWt)
            => weight - LearningRate * dTotRWt;
    }
}
