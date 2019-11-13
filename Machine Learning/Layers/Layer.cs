using System;
using System.Collections.Generic;

namespace ML.Layers
{
    public abstract class Layer
    {
        protected Layer(int neuronCount)
        {
            _neuronOutValues = new double[neuronCount];
            _neuronOutDerivatives = new double[neuronCount];
            _neuronDeltas = new double[neuronCount];
        }
        
        public int GetLayerCount() => 1 + (_next?.GetLayerCount() ?? 0);
        public int NeuronCount
        {
            get => _neuronOutValues.Length;
            set
            {
                if (_neuronOutValues.Length != value)
                    ResizeNeuronArrays(value, true);
            }
        }

        internal protected virtual void ResizeNeuronArrays(int neuronCount, bool notifyOwningNetwork)
        {
            Array.Resize(ref _neuronOutValues, neuronCount);
            Array.Resize(ref _neuronOutDerivatives, neuronCount);
            Array.Resize(ref _neuronDeltas, neuronCount);

            if (notifyOwningNetwork)
                OwningNetwork?.LayersChanged();
        }
        internal protected virtual void PreviousChanged()
        {

        }

        internal protected Layer _previous;
        internal protected Layer _next;

        protected double[] _neuronOutValues;
        protected double[] _neuronOutDerivatives;
        protected double[] _neuronDeltas;

        public Network OwningNetwork { get; private set; }
        public Dictionary<string, int> NeuronNameIndexAssocations { get; set; }

        public double[] NeuronOutValues
        {
            get => _neuronOutValues;
            set
            {
                _neuronOutValues = value ?? new double[0];
                NeuronCount = _neuronOutValues.Length;
            }
        }

        public virtual Layer Previous
        {
            get => _previous;
            set
            {
                if (_previous != null)
                    _previous._next = null;
                _previous = value;
                if (_previous != null)
                    _previous._next = this;
            }
        }
        public virtual Layer Next
        {
            get => _next;
            set
            {
                if (_next != null)
                {
                    _next._previous = null;
                }
                _next = value;
                if (_next != null)
                {
                    _next._previous = this;
                }
            }
        }

        public double[] NeuronOutDerivatives => _neuronOutDerivatives;

        public double? GetNeuronValue(string name)
        {
            if (NeuronNameIndexAssocations == null ||
                !NeuronNameIndexAssocations.ContainsKey(name))
                return null;

            int index = NeuronNameIndexAssocations[name];
            if (index >= 0 && index < _neuronOutValues.Length)
                return _neuronOutValues[index];

            return null;
        }

        public virtual Layer Initialize(Network owner, Random rand, bool recursive)
        {
            OwningNetwork = owner;

            if (recursive)
                return Next?.Initialize(owner, rand, true) ?? this;

            return this;
        }

        public Layer GetLast()
        {
            Layer layer = this;
            while (layer.Next != null)
                layer = layer.Next;
            return layer;
        }

        public abstract Layer Clone();

        public double GetNeuronDelta(int neuronIndex)
        {
            return _neuronDeltas[neuronIndex];
        }
        public void SetNeuronDelta(int neuronIndex, double cost)
        {
            _neuronDeltas[neuronIndex] = cost;
        }

        public abstract Layer Forward();
        public abstract void Backward();

        public abstract double GetDTotROut(int prevNeuronIndex);
    }
}
