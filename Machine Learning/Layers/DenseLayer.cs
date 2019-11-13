using ML.ActivationFunctions;
using System;
using System.Collections.Generic;

namespace ML.Layers
{
    public class DenseLayer : Layer
    {
        public DenseLayer(ActivationFunction activation, int neuronCount, bool useBias) : base(neuronCount)
        {
            _neuronBiases = new double[neuronCount];
            _prevNeuronBiases = new double[neuronCount];

            _weights = new double[0];
            _prevWeights = new double[0];

            UseBias = useBias;
            Activation = activation;
        }

        protected ActivationFunction _activation = new AF_ReLU();
        protected double[] _weights;
        protected double[] _prevWeights;
        protected double[] _neuronBiases;
        protected double[] _prevNeuronBiases;

        public double[] Weights => _weights;
        public double[] Biases => _neuronBiases;
        public bool UseBias { get; set; } = true;
        public ActivationFunction Activation
        {
            get => _activation;
            set => _activation = value ?? new AF_ReLU();
        }
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

        public override Layer Initialize(Network owner, Random rand, bool recursive)
        {
            int index;
            int prevNeuronCount = _previous?.NeuronOutValues?.Length ?? 0;
            for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
            {
                _neuronBiases[neuronIndex] = _prevNeuronBiases[neuronIndex] = rand.NextDouble();
                for (int prevNeuronIndex = 0; prevNeuronIndex < prevNeuronCount; ++prevNeuronIndex)
                {
                    index = neuronIndex * prevNeuronCount + prevNeuronIndex;
                    _weights[index] = _prevWeights[index] = rand.NextDouble();
                }
            }

            return base.Initialize(owner, rand, recursive);
        }

        protected internal override void ResizeNeuronArrays(int neuronCount, bool notifyOwningNetwork)
        {
            Array.Resize(ref _neuronBiases, neuronCount);
            Array.Resize(ref _prevNeuronBiases, neuronCount);

            base.ResizeNeuronArrays(neuronCount, false);

            RemakeWeights();
            _next?.PreviousChanged();

            if (notifyOwningNetwork)
                OwningNetwork?.LayersChanged();
        }
        internal protected override void PreviousChanged()
        {
            RemakeWeights();
        }
        protected virtual void RemakeWeights()
        {
            int count = Previous.NeuronOutValues.Length * _neuronOutValues.Length;

            Array.Resize(ref _weights, count);
            Array.Resize(ref _prevWeights, count);
        }
        public override Layer Clone()
        {
            DenseLayer layer = new DenseLayer(Activation, NeuronCount, UseBias);
            Array.Copy(NeuronOutValues, layer.NeuronOutValues, NeuronCount);
            return layer;
        }
        /// <summary>
        /// Gets or sets a weight between this and the previous layer.
        /// </summary>
        /// <param name="neuronIndex">The neuron in this layer the weight affects.</param>
        /// <param name="inputNeuronIndex">The neuron in the previous layer this weight affects.</param>
        /// <returns>The weight value.</returns>
        public double GetWeight(int neuronIndex, int inputNeuronIndex, bool previous)
        {
            //if (_weights == null ||
            //    _previous?._weights == null ||
            //    neuronIndex >= _weights.Length ||
            //    inputNeuronIndex >= _previous._weights.Length)
            //    return 0.0f;

            int index = _previous.NeuronOutValues.Length * neuronIndex + inputNeuronIndex;
            return previous ? _prevWeights[index] : _weights[index];
        }
        /// <summary>
        /// Gets or sets a weight between this and the previous layer.
        /// </summary>
        /// <param name="neuronIndex">The neuron in this layer the weight affects.</param>
        /// <param name="inputNeuronIndex">The neuron in the previous layer this weight affects.</param>
        /// <returns>The weight value.</returns>
        public void SetWeight(int neuronIndex, int inputNeuronIndex, double weight)
        {
            //if (_weights == null ||
            //    _previous?._weights == null ||
            //    neuronIndex >= _weights.Length ||
            //    inputNeuronIndex >= _previous._weights.Length)
            //    return;

            int index = _previous.NeuronOutValues.Length * neuronIndex + inputNeuronIndex;
            _prevWeights[index] = _weights[index];
            _weights[index] = weight;
        }
        public override Layer Forward()
        {
            if (_previous != null)
            {
                int prevNeuronCount = _previous.NeuronOutValues.Length;
                for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
                //Parallel.For(0, _neuronOutValues.Length, neuronIndex =>
                {
                    double sum = UseBias ? _neuronBiases[neuronIndex] : 0.0;
                    for (int prevNeuronIndex = 0; prevNeuronIndex < prevNeuronCount; ++prevNeuronIndex)
                    {
                        double weight = GetWeight(neuronIndex, prevNeuronIndex, false);
                        sum += _previous.NeuronOutValues[prevNeuronIndex] * weight;
                    }
                    _neuronOutDerivatives[neuronIndex] = Activation.Derivative(sum);
                    _neuronOutValues[neuronIndex] = Activation.Value(sum);
                }
                //);
            }
            return _next?.Forward() ?? this;
        }
        public override void Backward()
        {
            if (_previous == null)
                return;

            int totalConnections =
                _neuronOutValues.Length *
                _previous.NeuronOutValues.Length;

            //Parallel.For(0, totalConnections, conn =>
            for (int conn = 0; conn < totalConnections; ++conn)
            //for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
            //    for (int prevNeuronIndex = 0; prevNeuronIndex < _previous._neuronOutValues.Length; ++prevNeuronIndex)
            {
                int neuronIndex = conn / _previous.NeuronOutValues.Length;
                int prevNeuronIndex = conn % _previous.NeuronOutValues.Length;

                double dOutRNet = _neuronOutDerivatives[neuronIndex];
                double dNetRWt = _previous.NeuronOutValues[prevNeuronIndex];
                double dTotROut = _next is null ? GetNeuronDelta(neuronIndex) : _next.GetDTotROut(neuronIndex);

                double dTotRNet = dTotROut * dOutRNet;
                SetNeuronDelta(neuronIndex, dTotRNet);
                double dTotRWt = dTotRNet * dNetRWt;

                double weight = GetWeight(neuronIndex, prevNeuronIndex, false);
                double prevWeight = GetWeight(neuronIndex, prevNeuronIndex, true);

                weight = OwningNetwork.Optimizer.UpdateWeight(weight, prevWeight, dTotRWt);

                SetWeight(neuronIndex, prevNeuronIndex, weight);

                if (UseBias && prevNeuronIndex == 0)
                {
                    //Set bias for this neuron
                    double dTotRBias = dTotRNet * 1.0;
                    double bias = _neuronBiases[neuronIndex];
                    double prevBias = _prevNeuronBiases[neuronIndex];
                    _prevNeuronBiases[neuronIndex] = bias;

                    bias = OwningNetwork.Optimizer.UpdateWeight(bias, prevBias, dTotRBias);

                    _neuronBiases[neuronIndex] = bias;
                }
            }
            //);

            _previous.Backward();
        }
        public override double GetDTotROut(int prevNeuronIndex)
        {
            double dTotROut = 0.0;
            for (int nextNeuronIndex = 0; nextNeuronIndex < NeuronOutValues.Length; ++nextNeuronIndex)
            {
                double dNextTotROut = GetNeuronDelta(nextNeuronIndex);
                double dNextOutRNet = NeuronOutDerivatives[nextNeuronIndex];
                double dNextTotRNet = dNextTotROut * dNextOutRNet;
                double dNextNetROut = GetWeight(nextNeuronIndex, prevNeuronIndex, true);
                dTotROut += dNextTotRNet * dNextNetROut;
            }
            return dTotROut;
        }
    }
}
