namespace ML.CostFunctions
{
    public class CF_DiffSquared : CostFunction
    {
        public override double Evaluate(double target, double actual)
        {
            //Calc the distance between the value we have and the value we want
            double diff = target - actual;

            //Squaring makes larger errors cost more than smaller errors
            //Also makes any negative errors irrelevant, simply distance from 0
            //The 0.5 is added just to make the derivative come out clean, not necessary
            return 0.5f * diff * diff;
        }
        public override double Derivative(double target, double actual)
        {
            return actual - target;
        }
    }
}
