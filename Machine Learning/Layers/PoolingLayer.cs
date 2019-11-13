using System;

namespace ML.Layers
{
    public class PoolingLayer : Layer
    {
        public PoolingLayer(EPoolingType type, (int inputSize, int filterSize, int stride)[] dimensions) : base(0)
        {
            Type = type;
            Dimensions = dimensions;
            OutputSizes = new int[dimensions.Length];

            NeuronCount = CalcNeuronCount();
        }
        public enum EPoolingType
        {
            Max,
            Average,
        }

        public EPoolingType Type { get; set; }
        public (int inputSize, int filterSize, int stride)[] Dimensions { get; set; }
        public int[] OutputSizes { get; set; }

        private int CalcNeuronCount()
        {
            int count = 1;
            for (int i = 0; i < Dimensions.Length; ++i)
            {
                var (inputSize, filterSize, stride) = Dimensions[i];
                int newDim = inputSize / filterSize;
                OutputSizes[i] = newDim;
                count *= newDim;
            }
            return count;
        }

        public override Layer Clone()
        {
            PoolingLayer layer = new PoolingLayer(Type, Dimensions);
            Array.Copy(NeuronOutValues, layer.NeuronOutValues, NeuronCount);
            return layer;
        }
        public double GetInputValue(params int[] dimensionIndices)
        {
            int index = dimensionIndices[0];
            if (dimensionIndices.Length > 1)
            {
                int add = dimensionIndices[dimensionIndices.Length - 1];
                for (int dim = 0; dim < dimensionIndices.Length - 1; ++dim)
                    add *= Dimensions[dimensionIndices[dim]].inputSize;
                index += add;
            }
            return _previous.NeuronOutValues[index];
        }
        public override Layer Forward()
        {
            if (_previous != null)
            {
                var prevValues = _previous.NeuronOutValues;
                int[] strides = new int[Dimensions.Length];
                for (int dim = 0; dim < Dimensions.Length; ++dim)
                {
                    var (inputSize, filterSize, stride) = Dimensions[dim];
                    var samples = inputSize / filterSize;
                    for (int s = 0; s < samples; ++s)
                    {
                        for (int f = 0; f < filterSize; ++f)
                        {

                        }
                    }
                }
            }
            return _next?.Forward() ?? this;
        }
        public override void Backward() => throw new NotImplementedException();
        public override double GetDTotROut(int prevNeuronIndex) => throw new NotImplementedException();
    }
}
