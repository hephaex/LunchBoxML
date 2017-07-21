using System;
using System.Collections.Generic;
using System.Linq;

using Accord.Math;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Statistics.Kernels;
using Accord.Statistics.Filters;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using System.Data;
using Accord.MachineLearning;
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Models.Markov.Topology;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Networks;

namespace ProvingGround.MachineLearning.Classes
{
    /// <summary>
    /// Machine Learning Functions
    /// </summary>
    public class clsML
    {
        #region Regression
        /// <summary>
        /// Linear Regression
        /// </summary>
        /// <param name="test">Data to test</param>
        /// <param name="inputList">Learning inputs</param>
        /// <param name="outputList">Learning outputs</param>
        /// <returns>Result</returns>
        public double LinearRegression(double test, List<double> inputList, List<double> outputList)
        {

            // sample data
            double[] inputs = inputList.ToArray();
            double[] outputs = outputList.ToArray();

            // Use Ordinary Least Squares with simple linear regression
            OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
            SimpleLinearRegression regression = ols.Learn(inputs, outputs);

            double result = regression.Transform(test);

            return result;
        }

        /// <summary>
        /// Non Linear Regression
        /// </summary>
        /// <param name="test">Data to test</param>
        /// <param name="inputList">Learing inputs</param>
        /// <param name="outputList">Learning outputs</param>
        /// <param name="degree">Degree</param>
        /// <param name="complex">Complexity</param>
        /// <returns>Result</returns>
        public Tuple<double, double[], double> NonLinearRegression(List<double> test, List<List<double>> inputList, List<double> outputList, int degree, double complex)
        {

            // Training data
            double[][] inputs = inputList.Select(a => a.ToArray()).ToArray();
            double[] outputs = outputList.ToArray();
            double[] testdata = test.ToArray();

            Accord.Math.Random.Generator.Seed = 0;

            // Create the sequential minimal optimization teacher
            var learn = new SequentialMinimalOptimizationRegression<Polynomial>()
            {
                Kernel = new Polynomial(degree), // Polynomial Kernel of 2nd degree
                Complexity = complex
            };

            // Run the learning algorithm
            SupportVectorMachine<Polynomial> svm = learn.Learn(inputs, outputs);

            // Compute the predicted scores
            double[] predicted = svm.Score(inputs);

            // Compute the error between the expected and predicted
            double error = new SquareLoss(outputs).Loss(predicted);

            // Compute the answer for one particular example
            double fxy = svm.Score(testdata); // 1.0003849827673186
            return Tuple.Create<double, double[], double>(fxy, predicted, error);
        }

        /// <summary>
        /// Logistic Regression
        /// </summary>
        /// <param name="test">Data to test</param>
        /// <param name="inputList">Learning inputs</param>
        /// <param name="outputList">Learnout outputs</param>
        /// <returns>Result</returns>
        public Tuple<bool, double> LogisticRegression(List<double> test, List<List<double>> inputList, List<bool> outputList)
        {
            // Training data
            double[][] input = inputList.Select(a => a.ToArray()).ToArray();
            bool[] output = outputList.ToArray();
            double[] testdata = test.ToArray();


            // Create a new Iterative Reweighted Least Squares algorithm
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                Tolerance = 1e-4,
                MaxIterations = 100,
                Regularization = 0
            };

            // Now, we can use the learner to finally estimate our model:
            LogisticRegression regression = learner.Learn(input, output);

            double ageOdds = regression.GetOddsRatio(1);
            double smokeOdds = regression.GetOddsRatio(2);
            double score = regression.Probability(testdata);
            bool actual = regression.Decide(testdata);

            return Tuple.Create<bool, double>(actual, score);

        }

        /// <summary>
        /// Multivariate Linear Regression
        /// </summary>
        /// <param name="testList">List to test</param>
        /// <param name="inputList">Learning inputs</param>
        /// <param name="outputList">Learning outputs</param>
        /// <returns>Result</returns>
        public double[][] MultivariateLinearRegression(List<List<double>> testList, List<List<double>> inputList, List<List<double>> outputList)
        {

            // Training data
            double[][] inputs = inputList.Select(a => a.ToArray()).ToArray();
            double[][] outputs = outputList.Select(a => a.ToArray()).ToArray();
            double[][] testinputs = testList.Select(a => a.ToArray()).ToArray();

            // use ordinary least squares with multivariate linear regression
            OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
            MultivariateLinearRegression regression = ols.Learn(inputs, outputs);

            double[][] predictions = regression.Transform(testinputs);
            return predictions;

        }
        #endregion

        #region Classification
        /// <summary>
        /// Naive Bayes Classifier
        /// </summary>
        /// <param name="test">Data to test</param>
        /// <param name="data">All data</param>
        /// <param name="inputColumns">Input columns for data</param>
        /// <param name="outputColumn">Ouput column</param>
        /// <returns>Result</returns>
        public Tuple<string, int, double[]> NaiveBayesClassifier(List<string> test, DataTable data, List<string> inputColumns, string outputColumn)
        {

            List<string> colNames = new List<string>();
            foreach (DataColumn dc in data.Columns)
            {
                foreach (string input in inputColumns)
                {
                    if (input == dc.ColumnName)
                    {
                        colNames.Add(dc.ColumnName);
                    }
                }
                if (dc.ColumnName == outputColumn)
                {
                    colNames.Add(dc.ColumnName);
                }
            }
            string[] codes = colNames.ToArray();

            Codification codebook = new Codification(data, codes);

            // Extract input and output pairs to train
            DataTable symbols = codebook.Apply(data);
            int[][] inputs = symbols.ToJagged<int>(inputColumns.ToArray());
            int[] outputs = symbols.ToArray<int>(outputColumn);

            // Create a new Naive Bayes learning
            var learner = new NaiveBayesLearning();

            // Learn a Naive Bayes model from the examples
            NaiveBayes nb = learner.Learn(inputs, outputs);

            // Consider we would like to know whether one should play tennis at a
            // sunny, cool, humid and windy day. Let us first encode this instance
            int[] instance = codebook.Transform(test.ToArray());

            // Let us obtain the numeric output that represents the answer
            int c = nb.Decide(instance); // answer will be 0

            // Now let us convert the numeric output to an actual "Yes" or "No" answer
            string result = codebook.Revert(outputColumn, c); // answer will be "No"

            // We can also extract the probabilities for each possible answer
            double[] probs = nb.Probabilities(instance); // { 0.795, 0.205 }

            return Tuple.Create<string, int, double[]>(result, c, probs);
        }

        /// <summary>
        /// Gaussian Mixture Classifier
        /// </summary>
        /// <param name="inputList">Learning samples</param>
        /// <param name="components">Components</param>
        /// <returns>Result</returns>
        public Tuple<int[], double[], double[][]> GaussianMixture(List<List<double>> inputList, int components)
        {
            Accord.Math.Random.Generator.Seed = 0;

            // Test Samples
            double[][] samples = inputList.Select(a => a.ToArray()).ToArray();

            // Create a new Gaussian Mixture Model with 2 components
            GaussianMixtureModel gmm = new GaussianMixtureModel(components);

            // Estimate the Gaussian Mixture
            var clusters = gmm.Learn(samples);

            // Predict cluster labels for each sample
            int[] predicted = clusters.Decide(samples);
            double[] logLikelihoods = clusters.LogLikelihood(samples);
            double[][] probabilities = clusters.Probabilities(samples);

            return Tuple.Create<int[], double[], double[][]>(predicted, logLikelihoods, probabilities);
        }
        #endregion

        #region Networks
        /// <summary>
        /// Neural Network
        /// </summary>
        /// <param name="test">Test</param>
        /// <param name="inputList">Input list</param>
        /// <param name="labelList">Label list</param>
        /// <param name="hiddenNeurons">Hidden neurons</param>
        /// <param name="teach">teach</param>
        /// <returns>Result</returns>
        public string[] NeuralNetwork(List<List<double>> test, List<List<double>> inputList, List<int> labelList, int hiddenNeurons, int teach)
        {

            int[] labels = labelList.ToArray();
            double[][] input = inputList.Select(a => a.ToArray()).ToArray();
            double[][] testinput = test.Select(a => a.ToArray()).ToArray();

            int numberOfInputs = testinput[0].Length;
            int[] MyDistinctArray = labels.Distinct();
            int numberOfClasses = MyDistinctArray.Length;

            double[][] outputs = Accord.Math.Jagged.OneHot(labels);

            // Next we can proceed to create our network
            var function = new BipolarSigmoidFunction(2);
            var network = new ActivationNetwork(function,
              numberOfInputs, hiddenNeurons, numberOfClasses);

            // Heuristically randomize the network
            new NguyenWidrow(network).Randomize();

            // Create the learning algorithm
            var teacher = new LevenbergMarquardtLearning(network);

            // Teach the network for 10 iterations:
            double error = Double.PositiveInfinity;
            for (int i = 0; i < teach; i++)
                error = teacher.RunEpoch(input, outputs);

            // At this point, the network should be able to 
            // perfectly classify the training input points.
            List<string> strList = new List<string>();
            for (int i = 0; i < testinput.Length; i++)
            {
                int answer;
                double[] output = network.Compute(testinput[i]);
                double response = output.Max(out answer);

                int expected = labels[i];
                strList.Add(answer.ToString());

                // at this point, the variables 'answer' and
                // 'expected' should contain the same value.
            }
            string[] results = strList.ToArray();
            return results;

        }

        /// <summary>
        /// Restricted Boltzmann Soliver
        /// </summary>
        /// <param name="testList">Test list</param>
        /// <param name="inputList">Learning inputs</param>
        /// <param name="outputList">Learning outputs</param>
        /// <returns>Result</returns>
        public double[] RestrictedBoltzmann(List<double> testList, List<List<double>> inputList, List<List<double>> outputList)
        {

            // Training data
            double[][] inputs = inputList.Select(j => j.ToArray()).ToArray();
            double[][] outputs = outputList.Select(k => k.ToArray()).ToArray();
            double[] test = testList.ToArray();

            // Create a Bernoulli activation function
            var function = new BernoulliFunction(alpha: 0.5);


            int inputsCount = inputs[0].Length;
            int hiddenNeurons = outputs[0].Length;
            //int inputsCount = inputs[0].Length;
            //int hiddenNeurons = outputs[0].Length;
            // Create a Restricted Boltzmann Machine for 6 inputs and with 1 hidden neuron
            var rbm = new RestrictedBoltzmannMachine(function, inputsCount, hiddenNeurons);

            // Create the learning algorithm for RBMs
            var teacher = new ContrastiveDivergenceLearning(rbm)
            {
                Momentum = 0,
                LearningRate = 0.1,
                Decay = 0
            };

            // learn 5000 iterations
            for (int i = 0; i < 5000; i++)
                teacher.RunEpoch(inputs);

            // Compute the machine answers for the given inputs:
            double[] a = rbm.Compute(test);
            return a;
        }
        #endregion

        #region Models
        /// <summary>
        /// Hidden Markov Model
        /// </summary>
        /// <param name="inputList">Input list</param>
        /// <param name="num">Number</param>
        /// <returns>Result</returns>
        public string[] HiddenMarkovModel(List<List<string>> inputList, int num)
        {
            string[][] inputs = inputList.Select(a => a.ToArray()).ToArray();
            Accord.Math.Random.Generator.Seed = 42;

            var codebook = new Codification("Words", inputs);

            int[][] sequence = codebook.Transform("Words", inputs);

            var topology = new Forward(states: 4);
            int symbols = codebook["Words"].NumberOfSymbols;

            var hmm = new HiddenMarkovModel(topology, symbols);

            var teacher = new BaumWelchLearning(hmm);

            teacher.Learn(sequence);

            int[] sample = hmm.Generate(num);

            string[] result = codebook.Revert("Words", sample);

            return result;
        }
        #endregion
    }
}
