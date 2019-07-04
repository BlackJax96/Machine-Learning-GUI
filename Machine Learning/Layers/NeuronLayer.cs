using System;
using System.Collections.Generic;

namespace TheraEngine.Core.Maths.MachineLearning
{
    public class NeuronLayer
    {
        private NeuronLayer(ActivationFunction activation, int neuronCount, bool useBias)
        {
            Activation = activation;

            _neuronOutValues = new double[neuronCount];
            _neuronBiases = new double[neuronCount];
            _prevNeuronBiases = new double[neuronCount];
            _neuronOutDerivatives = new double[neuronCount];

            _weights = new double[0];
            _prevWeights = new double[0];
            _neuronDeltas = new double[0];

            UseBias = useBias;
        }
        //private NeuronLayer(params double[] values) : this(null, values?.Length ?? 0, false)
        //{
        //    if (values == null)
        //        return;
        //    for (int i = 0; i < values.Length; ++i)
        //        _neuronOutValues[i] = values[i];
        //}

        public static NeuronLayer Dense(ActivationFunction activation, int neuronCount, bool useBias)
            => new NeuronLayer(activation, neuronCount, useBias);

        //public static NeuronLayer Input(params double[] values)
        //    => new NeuronLayer(values);
        
        public static NeuronLayer Conv1D(ActivationFunction activation, int imageCount, int imageWidth, bool useBias)
            => new NeuronLayer(activation, imageCount * imageWidth, useBias);

        public static NeuronLayer Conv2D(ActivationFunction activation, int imageCount, int imageWidth, int imageHeight, bool useBias)
            => new NeuronLayer(activation, imageCount * imageWidth * imageHeight, useBias);

        public static NeuronLayer Conv3D(ActivationFunction activation, int imageCount, int imageWidth, int imageHeight, int imageDepth, bool useBias)
            => new NeuronLayer(activation, imageCount * imageWidth * imageHeight * imageDepth, useBias);
        
        public int GetLayerCount() => 1 + (_next?.GetLayerCount() ?? 0);

        private ActivationFunction _activation = new AF_ReLU();

        private NeuronLayer _previous;
        private NeuronLayer _next;
        
        private double[] _weights;
        private double[] _prevWeights;
        private double[] _neuronDeltas;
        
        private double[] _neuronOutValues;
        private double[] _neuronBiases;
        private double[] _prevNeuronBiases;
        private double[] _neuronOutDerivatives;

        public double[] Weights => _weights;
        public double[] Biases => _neuronBiases;

        public bool UseBias { get; set; } = true;
        public Network OwningNetwork { get; private set; }
        public Dictionary<string, int> NeuronNameIndexAssocations { get; set; }
        public ActivationFunction Activation
        {
            get => _activation;
            set => _activation = value ?? new AF_ReLU();
        }

        public double[] NeuronOutValues
        {
            get => _neuronOutValues;
            set
            {
                int prevCount = _neuronOutValues.Length;
                _neuronOutValues = value ?? new double[0];
                int nCount = _neuronOutValues.Length;
                if (nCount != prevCount)
                {
                    Array.Resize(ref _neuronBiases, nCount);
                    Array.Resize(ref _prevNeuronBiases, nCount);
                    Array.Resize(ref _neuronOutDerivatives, nCount);
                    RemakeWeights();
                    _next?.RemakeWeights();
                    OwningNetwork.LayersChanged();
                }
            }
        }
        
        public NeuronLayer Previous
        {
            get => _previous;
            set
            {
                if (_previous != null)
                    _previous._next = null;
                _previous = value;
                if (_previous != null)
                    _previous._next = this;
                RemakeWeights();
            }
        }
        public NeuronLayer Next
        {
            get => _next;
            set
            {
                if (_next != null)
                {
                    _next._previous = null;
                    _next.RemakeWeights();
                }
                _next = value;
                if (_next != null)
                {
                    _next._previous = this;
                    _next.RemakeWeights();
                }
            }
        }

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

        private void RemakeWeights()
        {
            int count = _previous._neuronOutValues.Length * _neuronOutValues.Length;
            Array.Resize(ref _weights, count);
            Array.Resize(ref _prevWeights, count);
            Array.Resize(ref _neuronDeltas, count);
        }

        public NeuronLayer Initialize(Network owner, Random rand, bool recursive)
        {
            OwningNetwork = owner;

            int index;
            int prevNeuronCount = _previous?._neuronOutValues?.Length ?? 0;
            for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
            {
                _neuronBiases[neuronIndex] = _prevNeuronBiases[neuronIndex] = rand.NextDouble();
                for (int prevNeuronIndex = 0; prevNeuronIndex < prevNeuronCount; ++prevNeuronIndex)
                {
                    index = neuronIndex * prevNeuronCount + prevNeuronIndex;
                    _weights[index] = _prevWeights[index] = rand.NextDouble();
                }
            }

            if (recursive)
                return Next?.Initialize(owner, rand, true) ?? this;

            return this;
        }

        public NeuronLayer GetLast()
        {
            NeuronLayer layer = this;
            while (layer.Next != null)
                layer = layer.Next;
            return layer;
        }

        public NeuronLayer FeedForward()
        {
            if (_previous != null)
            {
                int prevNeuronCount = _previous._neuronOutValues.Length;
                for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
                //Parallel.For(0, _neuronOutValues.Length, neuronIndex =>
                {
                    double sum = UseBias ? _neuronBiases[neuronIndex] : 0.0;
                    for (int prevNeuronIndex = 0; prevNeuronIndex < prevNeuronCount; ++prevNeuronIndex)
                    {
                        double weight = GetWeight(neuronIndex, prevNeuronIndex, false);
                        sum += _previous._neuronOutValues[prevNeuronIndex] * weight;
                    }
                    _neuronOutDerivatives[neuronIndex] = Activation.Derivative(sum);
                    _neuronOutValues[neuronIndex] = Activation.Value(sum);
                }
                //);
            }
            return _next?.FeedForward() ?? this;
        }
        public void BackPropagate(double learningRate, double momentum)
        {
            if (_previous == null)
                return;

            int totalConnections =
                _neuronOutValues.Length *
                _previous._neuronOutValues.Length;

            //Parallel.For(0, totalConnections, conn =>
            for (int conn = 0; conn < totalConnections; ++conn)
            //for (int neuronIndex = 0; neuronIndex < _neuronOutValues.Length; ++neuronIndex)
            //    for (int prevNeuronIndex = 0; prevNeuronIndex < _previous._neuronOutValues.Length; ++prevNeuronIndex)
                {
                    int neuronIndex = conn / _previous._neuronOutValues.Length;
                    int prevNeuronIndex = conn % _previous._neuronOutValues.Length;

                    double dTotROut;
                    double dOutRNet = _neuronOutDerivatives[neuronIndex];
                    double dNetRWt = _previous._neuronOutValues[prevNeuronIndex];
                    if (_next == null)
                    {
                        dTotROut = GetNeuronDelta(neuronIndex);
                    }
                    else
                    {
                        dTotROut = 0.0;
                        for (int nextNeuronIndex = 0; nextNeuronIndex < _next._neuronOutValues.Length; ++nextNeuronIndex)
                        {
                            double dNextTotROut = _next.GetNeuronDelta(nextNeuronIndex);
                            double dNextOutRNet = _next._neuronOutDerivatives[nextNeuronIndex];
                            double dNextTotRNet = dNextTotROut * dNextOutRNet;
                            double dNextNetROut = _next.GetWeight(nextNeuronIndex, neuronIndex, true);
                            dTotROut += dNextTotRNet * dNextNetROut;
                        }
                    }

                    double dTotRNet = dTotROut * dOutRNet;
                    SetNeuronDelta(neuronIndex, dTotRNet);
                    double dTotRWt = dTotRNet * dNetRWt;

                    double weight = GetWeight(neuronIndex, prevNeuronIndex, false);
                    double prevWeight = GetWeight(neuronIndex, prevNeuronIndex, true);
                    weight = weight - learningRate * dTotRWt + momentum * (weight - prevWeight);
                    SetWeight(neuronIndex, prevNeuronIndex, weight);

                    if (UseBias && prevNeuronIndex == 0)
                    {
                        //Set bias for this neuron
                        double dTotRBias = dTotRNet * 1.0;
                        double bias = _neuronBiases[neuronIndex];
                        double prevBias = _prevNeuronBiases[neuronIndex];
                        _prevNeuronBiases[neuronIndex] = bias;
                        bias -= learningRate * dTotRBias + momentum * (bias - prevBias);
                        _neuronBiases[neuronIndex] = bias;
                    }
                }
            //);

            _previous.BackPropagate(learningRate, momentum);
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

            int index = _previous._neuronOutValues.Length * neuronIndex + inputNeuronIndex;
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

            int index = _previous._neuronOutValues.Length * neuronIndex + inputNeuronIndex;
            _prevWeights[index] = _weights[index];
            _weights[index] = weight;
        }
        /// <summary>
        /// Gets or sets a weight between this and the previous layer.
        /// </summary>
        /// <param name="neuronIndex">The neuron in this layer the weight affects.</param>
        /// <param name="inputNeuronIndex">The neuron in the previous layer this weight affects.</param>
        /// <returns>The weight value.</returns>
        public double GetNeuronDelta(int neuronIndex)
        {
            //if (_weightDeltas == null ||
            //    _previous?._weightDeltas == null ||
            //    neuronIndex >= _weightDeltas.Length ||
            //    inputNeuronIndex >= _previous._weightDeltas.Length)
            //    return 0.0f;

            return _neuronDeltas[neuronIndex];
        }
        /// <summary>
        /// Gets or sets a weight between this and the previous layer.
        /// </summary>
        /// <param name="neuronIndex">The neuron in this layer the weight affects.</param>
        /// <param name="inputNeuronIndex">The neuron in the previous layer this weight affects.</param>
        /// <returns>The weight value.</returns>
        public void SetNeuronDelta(int neuronIndex, double cost)
        {
            //if (_weightDeltas == null ||
            //    _previous?._weightDeltas == null ||
            //    neuronIndex >= _weightDeltas.Length ||
            //    inputNeuronIndex >= _previous._weightDeltas.Length)
            //    return;

            _neuronDeltas[neuronIndex] = cost;
        }
    }
}
