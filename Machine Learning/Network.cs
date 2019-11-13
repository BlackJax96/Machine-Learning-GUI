using CsvHelper;
using CsvHelper.Configuration;
using ML.ActivationFunctions;
using ML.CostFunctions;
using ML.Layers;
using ML.Optimizers;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ML
{
    public enum EErrorTrainingType
    {
        Individual,
        IndividualWeighted,
        Total,
    }
    public delegate void DelForwardPropagated(double[] output);
    public delegate void DelCostChanged(double oldCost, double newCost, int iteration);
    public class Network : IEnumerable<Layer>
    {
        public Network() { }
        public Network(int inputCount, CostFunction costFunc, Optimizer optimizer, params Layer[] layers)
        {
            CostFunction = costFunc;
            Optimizer = optimizer;

            Layer first = new DenseLayer(null, inputCount, false);
            if (layers == null || layers.Length == 0)
                return;
            for (int i = 1; i < layers.Length; ++i)
                layers[i].Previous = layers[i - 1];
            first.Next = layers[0];
            Input = first;
        }

        public event Action BackPropagated;
        public event DelForwardPropagated ForwardPropagated;
        public event DelCostChanged CostChanged;

        public double ConfidencePercentage => (1.0f - _currentCost) * 100.0f;

        private double _currentCost;
        private Layer _input;

        /// <summary>
        /// The method to use to calculate the cost of an output neuron.
        /// </summary>
        public CostFunction CostFunction { get; set; } = new CF_DiffSquared();
        /// <summary>
        /// 
        /// </summary>
        public Optimizer Optimizer { get; set; } = new O_Momentum();
        /// <summary>
        /// This is the first layer in the network, used solely for input.
        /// All layers after this one are either hidden layers or the output layer.
        /// Layers attached before this one are not used.
        /// </summary>
        public Layer Input
        {
            get => _input;
            set
            {
                _input = value;
                if (_input != null)
                {
                    Random random = new Random();
                    Layer layer = _input;
                    layer.Initialize(this, random, false);
                    while (layer.Next != null)
                    {
                        layer = layer.Next;
                        layer.Initialize(this, random, false);
                    }
                    Output = layer;
                    LayersChanged();
                }
            }
        }
        public Layer Output { get; private set; }

        internal void LayersChanged()
        {
            LayerCount = Input?.GetLayerCount() ?? 0;
        }

        /// <summary>
        /// Recursively counts the attached layers.
        /// </summary>
        public int LayerCount { get; private set; }

        public double PreviousCost { get; private set; }
        public double CurrentCost
        {
            get => _currentCost;
            private set
            {
                PreviousCost = _currentCost;
                _currentCost = value;
                CostChanged?.Invoke(PreviousCost, _currentCost, TotalIterationsTrained);
            }
        }

        public int TotalIterationsTrained { get; private set; } = 0;

        /// <summary>
        /// Propagates inputs through the network and returns the corresponding output values.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] Calculate(params double[] input)
        {
            if (Input == null)
                throw new InvalidOperationException($"{nameof(Input)} cannot be null.");
            if (Input.NeuronOutValues == null)
                throw new InvalidOperationException($"{nameof(Input)}.{nameof(DenseLayer.NeuronOutValues)} cannot be null.");
            if (input == null)
                throw new InvalidOperationException($"{nameof(input)} cannot be null.");
            if (Input.NeuronOutValues.Length != input.Length)
                throw new InvalidOperationException($"{nameof(input)}.{nameof(DenseLayer.NeuronOutValues)} count does not match {nameof(Input)} count.");

            Input.NeuronOutValues = input;

            Layer output = Input.Forward();

            return output.NeuronOutValues;
        }
        /// <summary>
        /// Propagates inputs through the network and returns the corresponding output values.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] CalculateError(double[] input, double[] expectedOutput, out double totalError)
        {
            if (Input == null)
                throw new InvalidOperationException($"{nameof(Input)} cannot be null.");
            if (Input.NeuronOutValues == null)
                throw new InvalidOperationException($"{nameof(Input)}.{nameof(DenseLayer.NeuronOutValues)} cannot be null.");
            if (input == null)
                throw new InvalidOperationException($"{nameof(input)} cannot be null.");
            if (Input.NeuronOutValues.Length != input.Length)
                throw new InvalidOperationException($"{nameof(input)}.{nameof(DenseLayer.NeuronOutValues)} count does not match {nameof(Input)} count.");

            Input.NeuronOutValues = input;

            Layer output = Input.Forward();

            totalError = 0.0;
            double[] error = new double[output.NeuronOutValues.Length];
            for (int i = 0; i < output.NeuronOutValues.Length; ++i)
                totalError += error[i] = expectedOutput[i] - output.NeuronOutValues[i];

            return error;
        }
        public void Train(
            int iterations,
            params (double[] Input, double[] Output)[] trainingSets)
        {
            if (trainingSets == null || trainingSets.Length == 0)
                throw new InvalidOperationException($"Network needs {nameof(trainingSets)} to be set to train against.");
            if (Input == null)
                throw new InvalidOperationException($"Network needs {nameof(Input)} to be set to forward propagate through layers.");
            if (Input.Next == null)
                throw new InvalidOperationException($"Network needs at least two layers (set {nameof(Input)}.{nameof(DenseLayer.Next)}).");
            if (iterations <= 0)
                throw new InvalidOperationException($"{nameof(iterations)} must be greater than zero.");
            if (trainingSets.Any(x => 
                x.Input.Length != Input.NeuronOutValues.Length || 
                x.Output.Length != Output.NeuronOutValues.Length))
                throw new InvalidOperationException($"One or more sets in {nameof(trainingSets)} do not have the same number of input and output values as the first or last layer in the network.");

            double[] expectedOutput;
            int setIndex;
            double totalCost, targetValue, actualValue, neuronCost, neuronCostDeriv;
            for (int iteration = 0; iteration < iterations; ++iteration, ++TotalIterationsTrained)
            {
                setIndex = iteration % trainingSets.Length;
                expectedOutput = trainingSets[setIndex].Output;
                Input.NeuronOutValues = trainingSets[setIndex].Input;
                Input.Forward();

                totalCost = 0.0;
                for (int outputNeuronIndex = 0; outputNeuronIndex < Output.NeuronOutValues.Length; ++outputNeuronIndex)
                {
                    targetValue = expectedOutput[outputNeuronIndex];
                    actualValue = Output.NeuronOutValues[outputNeuronIndex];
                    neuronCost = CostFunction.Evaluate(targetValue, actualValue);
                    neuronCostDeriv = CostFunction.Derivative(targetValue, actualValue);
                    Output.SetNeuronDelta(outputNeuronIndex, neuronCostDeriv);
                    totalCost += neuronCost;
                }

                CurrentCost = totalCost;

                ForwardPropagated?.Invoke(Output.NeuronOutValues);

                Output.Backward();
                
                BackPropagated?.Invoke();
            }
        }
        public void Train(
            double targetError,
            EErrorTrainingType errorType,
            bool trainRandomly,
            params (double[] Input, double[] Output)[] trainingSets)
        {
            if (trainingSets == null || trainingSets.Length == 0)
                throw new InvalidOperationException($"Network needs {nameof(trainingSets)} to be set to train against.");
            if (Input == null)
                throw new InvalidOperationException($"Network needs {nameof(Input)} to be set to forward propagate through layers.");
            if (Input.Next == null)
                throw new InvalidOperationException($"Network needs at least two layers (set {nameof(Input)}.{nameof(DenseLayer.Next)}).");
            if (targetError <= 0)
                throw new InvalidOperationException($"{nameof(targetError)} must be greater than zero.");
            if (trainingSets.Any(x =>
                x.Input.Length != Input.NeuronOutValues.Length ||
                x.Output.Length != Output.NeuronOutValues.Length))
                throw new InvalidOperationException($"One or more sets in {nameof(trainingSets)} do not have the same number of input and output values as the first or last layer in the network.");

            double[] expectedOutput;
            int setIndex;
            double totalCost, targetValue, actualValue, neuronCost, neuronCostDeriv;
            int iteration = 0;
            double[] indivError = new double[Output.NeuronOutValues.Length];
            Random rand = new Random();
            
            Func<double, double, double, double> collectIndivError;
            Func<double, double[], bool> checkTrainingSuccess;
            Func<int, (double[], double[])[], Random, int> getTrainingSet = trainRandomly ? 
                (Func<int, (double[], double[])[], Random, int>)GetTrainingSetRandomly : 
                GetTrainingSetInOrder;

            switch (errorType)
            {
                default:
                case EErrorTrainingType.Total:
                    collectIndivError = CollectIndivErrorWeighted;
                    checkTrainingSuccess = CheckTrainingSuccessTotal;
                    break;
                case EErrorTrainingType.Individual:
                    collectIndivError = CollectIndivErrorUnweighted;
                    checkTrainingSuccess = CheckTrainingSuccessIndividual;
                    break;
                case EErrorTrainingType.IndividualWeighted:
                    collectIndivError = CollectIndivErrorWeighted;
                    checkTrainingSuccess = CheckTrainingSuccessIndividual;
                    break;
            }

            do
            {
                setIndex = getTrainingSet(iteration, trainingSets, rand);
                var set = trainingSets[setIndex];
                expectedOutput = set.Output;
                Input.NeuronOutValues = set.Input;
                Input.Forward();

                totalCost = 0.0;
                for (int outputNeuronIndex = 0; outputNeuronIndex < Output.NeuronOutValues.Length; ++outputNeuronIndex)
                {
                    targetValue = expectedOutput[outputNeuronIndex];
                    actualValue = Output.NeuronOutValues[outputNeuronIndex];
                    
                    neuronCost = CostFunction.Evaluate(targetValue, actualValue);
                    indivError[outputNeuronIndex] = collectIndivError(targetValue, actualValue, neuronCost);

                    neuronCostDeriv = CostFunction.Derivative(targetValue, actualValue);
                    Output.SetNeuronDelta(outputNeuronIndex, neuronCostDeriv);
                    totalCost += neuronCost;
                }

                CurrentCost = totalCost;

                //Update output display here
                ForwardPropagated?.Invoke(Output.NeuronOutValues);

                if (checkTrainingSuccess(targetError, indivError))
                    break;

                Output.Backward();
                ++TotalIterationsTrained;
                ++iteration;
            }
            while (true);
        }
        
        private int GetTrainingSetInOrder(int iteration, (double[] Input, double[] Output)[] trainingSets, Random rand)
            => iteration % trainingSets.Length;
        private int GetTrainingSetRandomly(int iteration, (double[] Input, double[] Output)[] trainingSets, Random rand)
            => rand.Next(0, trainingSets.Length);
        private double CollectIndivErrorUnweighted(double targetValue, double actualValue, double neuronCost)
            => Math.Abs(targetValue - actualValue);
        private double CollectIndivErrorWeighted(double targetValue, double actualValue, double neuronCost)
            => neuronCost;
        private bool CheckTrainingSuccessIndividual(double targetError, double[] indivError)
            => !indivError.Any(x => x > targetError);
        private bool CheckTrainingSuccessTotal(double targetError, double[] indivError)
            => Math.Abs(CurrentCost) < targetError;
        
        public Network Clone()
            => new Network
            {
                CostFunction = CostFunction,
                Input = Input.Clone()
            };
        
        //public void ToCSV(string path)
        //{
        //    using (CsvWriter writer = new CsvWriter(new StreamWriter(path, false)))
        //    {
        //        writer.WriteField(Input.NeuronOutValues.Length.ToString(CultureInfo.InvariantCulture));
        //        writer.WriteField(CostFunction.GetType().AssemblyQualifiedName.Replace(",", "|"));
        //        writer.WriteField(CurrentCost.ToString(CultureInfo.InvariantCulture));
        //        writer.NextRecord();

        //        Layer layer = Input;
        //        while (layer != null)
        //        {
        //            writer.WriteField(layer.Activation.GetType().AssemblyQualifiedName.Replace(",", "|"));
        //            writer.WriteField(layer.Weights.Length.ToString(CultureInfo.InvariantCulture));
        //            writer.WriteField(layer.Biases.Length.ToString(CultureInfo.InvariantCulture));
        //            writer.WriteField(layer.UseBias);

        //            for (int i = 0; i < layer.Weights.Length; ++i)
        //                writer.WriteField(layer.Weights[i].ToString(CultureInfo.InvariantCulture));

        //            for (int i = 0; i < layer.Biases.Length; ++i)
        //                writer.WriteField(layer.Biases[i].ToString(CultureInfo.InvariantCulture));

        //            writer.NextRecord();
        //            layer = layer.Next;
        //        }
        //    }
        //}
        //public static Network FromCSV(string path)
        //{
        //    Network net = new Network();
        //    Configuration config = new Configuration();
        //    using (CsvReader reader = new CsvReader(new StreamReader(path), config))
        //    {
        //        int inputCount = reader.GetField<int>(0);
        //        string costTypeStr = reader.GetField(1).Replace("|", ",");
        //        double cost = reader.GetField<double>(2);

        //        Type costType = Type.GetType(costTypeStr);
        //        net.CostFunction = Activator.CreateInstance(costType) as CostFunction;

        //        Layer prev = new Layer(null, inputCount, false);

        //        net.Input = prev;
        //        net.CurrentCost = cost;

        //        while (reader.Read())
        //        {
        //            string activTypeStr = reader.GetField(0).Replace("|", ",");
        //            int weightCount = reader.GetField<int>(1);
        //            int biasCount = reader.GetField<int>(2);
        //            bool useBias = reader.GetField<bool>(3);
                    
        //            Type activType = Type.GetType(activTypeStr);
        //            ActivationFunction activFunc = Activator.CreateInstance(activType) as ActivationFunction;

        //            Layer layer = new Layer(activFunc, biasCount, useBias);

        //            for (int i = 0; i < weightCount; ++i)
        //                layer.Weights[i] = reader.GetField<double>(i + 4);

        //            for (int i = 0; i < biasCount; ++i)
        //                layer.Biases[i] = reader.GetField<double>(i + 4 + weightCount);

        //            prev.Next = layer;
        //            prev = layer;
        //        }
        //    }
        //    return net;
        //}

        public IEnumerator<Layer> GetEnumerator()
        {
            Layer layer = _input;
            while (layer != null)
            {
                yield return layer;
                layer = layer.Next;
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
