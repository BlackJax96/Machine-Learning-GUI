using ML.ActivationFunctions;
using System;
using System.Collections.Generic;

namespace ML.Layers
{
    public class ScaleLayer : Layer
    {
        public ScaleLayer(int neuronCount) : base(neuronCount)
        {

        }

        public double Scale { get; set; }

        public override Layer Previous
        {
            get => _previous;
            set
            {
                if (_previous != null)
                    _previous._next = null;
                _previous = value;
                if (_previous != null)
                    _previous._next = this;
                PreviousChanged();
            }
        }
        public override Layer Next
        {
            get => base.Next;
            set
            {
                if (_next != null)
                {
                    _next._previous = null;
                    _next.PreviousChanged();
                }
                base.Next = value;
                if (_next != null)
                {
                    _next._previous = this;
                    _next.PreviousChanged();
                }
            }
        }

        public override Layer Clone()
        {
            ScaleLayer layer = new ScaleLayer(NeuronCount);
            Array.Copy(NeuronOutValues, layer.NeuronOutValues, NeuronCount);
            return layer;
        }
        public override Layer Forward()
        {
            if (_previous != null)
            {
                int prevNeuronCount = _previous.NeuronCount;
                if (prevNeuronCount != NeuronCount)
                    throw new InvalidOperationException("Scale layer needs the same amount of neurons as the previous layer.");

                for (int i = 0; i < NeuronCount; ++i)
                    NeuronOutValues[i] = _previous.NeuronOutValues[i] * Scale;
            }
            return _next?.Forward() ?? this;
        }
        public override void Backward()
        {
            if (_previous == null)
                return;

            int prevNeuronCount = _previous.NeuronCount;
            if (prevNeuronCount != NeuronCount)
                throw new InvalidOperationException("Scale layer needs the same amount of neurons as the previous layer.");

            for (int i = 0; i < NeuronCount; ++i)
                _previous.NeuronOutValues[i] = NeuronOutValues[i] / Scale;

            _previous.Backward();
        }
        public override double GetDTotROut(int prevNeuronIndex)
        {
            return 0.0;
        }
    }
}
