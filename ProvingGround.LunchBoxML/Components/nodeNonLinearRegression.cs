using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Nonlinear Regression Node
    /// </summary>
    public class nodeSeqMinimalRegression : GH_Component
    {
        #region Register Node
        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeSeqMinimalRegression()
            : base("Nonlinear Regression", "NonlineReg", "Solver for nonlinear regression problems using Sequential Minimal Optimization.", "LunchBox", "Machine Learning")
        {

        }

        /// <summary>
        /// Component Exposure
        /// </summary>
        public override GH_Exposure Exposure
        {
            get { return GH_Exposure.primary; }
        }

        /// <summary>
        /// GUID generator http://www.guidgenerator.com/online-guid-generator.aspx
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("fd9ab1cd-dff2-48c0-9ac3-6ebe7c24a992"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_NonLinearRegression; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter("Test Data", "Test", "Tree of data to test against learning data.", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Inputs", "Inputs", "Tree list of inputs.", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Output", "Output", "The list of Outputs.", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Sigma", "Sigma", "Sigma of prediction curve", GH_ParamAccess.item, 2);
            pManager.AddNumberParameter("Complexity", "Complex", "Complexity of the prediction", GH_ParamAccess.item, 100);
            pManager.AddIntegerParameter("Random Seed", "Seed", "Seed for number generator.", GH_ParamAccess.item, 5);
        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Resultant prediction");
            pManager.Register_GenericParam("Score", "Score", "The predicted scores");
            pManager.Register_GenericParam("Error", "Error", "Error between the expected and predicted");
        }
        #endregion

        #region Solution
        /// <summary>
        /// Code by the component
        /// </summary>
        /// <param name="DA"></param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // Tree Structure Input Variables
            GH_Structure<GH_Number> test = new GH_Structure<GH_Number>(); ;
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            List<double> outputs = new List<double>();
            int degree = 2;
            double complex = 100;
            int seed = 5;

            //Tree Variables
            DA.GetDataTree<GH_Number>(0, out test);
            DA.GetDataTree<GH_Number>(1, out inputs);
            DA.GetDataList(2, outputs);
            DA.GetData(3, ref degree);
            DA.GetData(4, ref complex);
            DA.GetData(5, ref seed);

            // list of lists
            List<List<double>> inputList = new List<List<double>>();
            List<List<double>> testList = new List<List<double>>();

            // input list of lists from tree
            for (int i = 0; i < inputs.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = inputs.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                inputList.Add(list);
            }

            // test list of lists from tree
            for (int i = 0; i < test.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = test.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                testList.Add(list);
            }

            //Result
            clsML learning = new clsML();
            Tuple<double[], double[], double> result = learning.SequentialMinimalOptimizationRegressionModel(testList, inputList, outputs, degree, complex, seed);

            //Output
            DA.SetDataList(0, result.Item2.ToList());
            DA.SetDataList(1, result.Item1.ToList());
            DA.SetData(2, result.Item3);
        }
        #endregion
    }  
}



