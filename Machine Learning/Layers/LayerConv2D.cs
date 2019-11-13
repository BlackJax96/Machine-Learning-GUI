using System;
using ML.ActivationFunctions;

namespace ML.Layers
{
    public class LayerConv2D : Layer
    {
        public LayerConv2D((int inputSize, int filterSize, int stride, int paddingSize)[] dimensions) : base(0)
        {
            Dimensions = dimensions;

            CalcFilterMatrixSize();
            NeuronCount = CalcNeuronCount();
        }

        public double[] FilterMatrixFlat { get; private set; }
        public (int inputSize, int filterSize, int stride, int paddingSize)[] Dimensions { get; set; }

        private void CalcFilterMatrixSize()
        {
            int size = 1;
            for (int i = 0; i < Dimensions.Length; ++i)
                size *= Dimensions[i].filterSize;
            FilterMatrixFlat = new double[size];
        }

        private int CalcNeuronCount()
        {
            int count = 0;

            return count;
        }

        public override Layer Initialize(Network owner, Random rand, bool recursive)
        {
            for (int i = 0; i < FilterMatrixFlat.Length; ++i)
                FilterMatrixFlat[i] = rand.NextDouble();

            return base.Initialize(owner, rand, recursive);
        }

        public override Layer Clone() => throw new NotImplementedException();
        public override Layer Forward()
        {
            if (_previous != null)
            {

            }
            return _next?.Forward() ?? this;
        }
        public override void Backward() => throw new NotImplementedException();
        public override double GetDTotROut(int prevNeuronIndex) => throw new NotImplementedException();
    }
}
