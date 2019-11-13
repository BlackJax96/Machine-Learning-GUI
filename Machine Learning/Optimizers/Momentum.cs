using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Optimizers
{
    public class O_Momentum : O_StaticLearningRate
    {
        public double Momentum { get; set; }

        public override double UpdateWeight(double weight, double prevWeight, double dTotRWt) 
            => weight - LearningRate * dTotRWt + Momentum * (weight - prevWeight);
    }
}
