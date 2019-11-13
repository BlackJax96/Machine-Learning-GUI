using ML;
using ML.ActivationFunctions;
using ML.CostFunctions;
using ML.Extensions;
using ML.Layers;
using ML.Optimizers;
using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace MLUI
{
    public static class Tests
    {
        public static event DelCostChanged CostChanged;
        private enum ELogicalOpType
        {
            OR,
            AND,
            XOR,
            NOR,
            XNOR,
            NAND,
        }
        public static async Task RunAll()
        {
            double learningRate = 0.8;
            double targetError = 0.004;
            double momentum = 0.2;
            int fracDigits = 0;

            await TrainLogicOp(ELogicalOpType.OR, learningRate, momentum, targetError, fracDigits);
            await TrainLogicOp(ELogicalOpType.AND, learningRate, momentum, targetError, fracDigits);
            await TrainLogicOp(ELogicalOpType.XOR, learningRate, momentum, targetError, fracDigits);
            await TrainLogicOp(ELogicalOpType.NOR, learningRate, momentum, targetError, fracDigits);
            await TrainLogicOp(ELogicalOpType.XNOR, learningRate, momentum, targetError, fracDigits);
            await TrainLogicOp(ELogicalOpType.NAND, learningRate, momentum, targetError, fracDigits);
        }

        private static async Task TrainLogicOp(
            ELogicalOpType type,
            double learningRate,
            double momentum,
            double targetError,
            int outputRoundingFracDigits)
        {
            (double[] Input, double[] Output)[] trainingSets = GetLogicOpInputOutput(type);

            Console.WriteLine($"Training {type.ToString()} network.");

            Network nw = new Network(2,
                new CF_DiffSquared(),
                new O_Momentum(),
                new DenseLayer(new AF_Sine(), 2, true),
                new DenseLayer(new AF_Logistic(), 1, true));

            Stopwatch timer = new Stopwatch();

            nw.CostChanged += CostChangedMethod;
            timer.Start();

            await Task.Run(() => nw.Train(targetError, EErrorTrainingType.Individual, false, trainingSets));

            timer.Stop();
            nw.CostChanged -= CostChangedMethod;

            Console.WriteLine($"{nw.TotalIterationsTrained} iterations finished in {timer.Elapsed.TotalSeconds} seconds.");

            //double totalError;

            double[] zz = nw.Calculate(trainingSets[0].Input); //nw.CalculateError(inputOutput[0].Item1, inputOutput[0].Item2, out totalError);
            Console.WriteLine("[0, 0]: " + zz.ToStringList(", ", o => Math.Round(o, outputRoundingFracDigits).ToString()));

            double[] oz = nw.Calculate(trainingSets[1].Input); //nw.CalculateError(inputOutput[1].Item1, inputOutput[1].Item2, out totalError);
            Console.WriteLine("[1, 0]: " + oz.ToStringList(", ", o => Math.Round(o, outputRoundingFracDigits).ToString()));

            double[] zo = nw.Calculate(trainingSets[2].Input); //nw.CalculateError(inputOutput[2].Item1, inputOutput[2].Item2, out totalError);
            Console.WriteLine("[0, 1]: " + zo.ToStringList(", ", o => Math.Round(o, outputRoundingFracDigits).ToString()));

            double[] oo = nw.Calculate(trainingSets[3].Input); //nw.CalculateError(inputOutput[3].Item1, inputOutput[3].Item2, out totalError);
            Console.WriteLine("[1, 1]: " + oo.ToStringList(", ", o => Math.Round(o, outputRoundingFracDigits).ToString()));
        }

        private static void CostChangedMethod(double oldCost, double newCost, int iteration)
        {
            CostChanged?.Invoke(oldCost, newCost, iteration);
        }

        private static (double[] Input, double[] Output)[] GetLogicOpInputOutput(ELogicalOpType type)
        {
            (double[], double[])[] inputOutput = new (double[], double[])[4];
            switch (type)
            {
                case ELogicalOpType.XOR:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 1.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 0.0 });
                    break;
                case ELogicalOpType.XNOR:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 0.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 1.0 });
                    break;
                case ELogicalOpType.AND:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 0.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 1.0 });
                    break;
                case ELogicalOpType.NOR:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 0.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 0.0 });
                    break;
                case ELogicalOpType.OR:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 0.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 1.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 1.0 });
                    break;
                case ELogicalOpType.NAND:
                    inputOutput[0] = (new double[] { 0.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[1] = (new double[] { 0.0, 1.0 }, new double[] { 1.0 });
                    inputOutput[2] = (new double[] { 1.0, 0.0 }, new double[] { 1.0 });
                    inputOutput[3] = (new double[] { 1.0, 1.0 }, new double[] { 0.0 });
                    break;
            }
            return inputOutput;
        }
    }
}
