using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Data;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Multivariate Linear Regression Node
    /// </summary>
    public class nodeMultivariateLinearRegression : GH_Component
    {
        #region Register Node
        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeMultivariateLinearRegression()
            : base("Multivariate Linear Regression", "MultiLineReg", "Solver for multivariate linear regression problems.", "LunchBox", "Machine Learning")
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
            get { return new Guid("20388bae-18d1-4a7a-b9bf-0e6fd378e06a"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_MVLinearRegression; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter ("Test Data", "Test", "Data to test against learning data.", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Inputs", "Inputs", "The list of inputs.", GH_ParamAccess.tree);
            pManager.AddNumberParameter("Output", "Output", "The list of Outputs.", GH_ParamAccess.tree);
        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Resultant prediction");
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
            GH_Structure<GH_Number> test = new GH_Structure<GH_Number>();
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            GH_Structure<GH_Number> outputs = new GH_Structure<GH_Number>();

            //Tree Variables
            DA.GetDataTree<GH_Number>(0, out test);
            DA.GetDataTree<GH_Number>(1, out inputs);
            DA.GetDataTree<GH_Number>(2, out outputs);

            // list of lists
            List<List<double>> testList = new List<List<double>>();
            List<List<double>> inputList = new List<List<double>>();
            List<List<double>> outputList = new List<List<double>>();

            // Test list of lists from tree
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

            // output list of lists from tree
            for (int i = 0; i < outputs.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = outputs.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                outputList.Add(list);
            }

            //Result
            clsML learning = new clsML();
            double[][] result = learning.MultivariateLinearRegression(testList, inputList, outputList);

            List<List<double>> resultList = result
                .Where(inner => inner != null) // Cope with uninitialised inner arrays.
                .Select(inner => inner.ToList()) // Project each inner array to a List<string>
                .ToList();

            DataTree<double> resultTree = new DataTree<double>();
            for (int i = 0; i < resultList.Count; i++)
            {
                for (int j = 0; j < resultList[i].Count; j++)
                {
                    GH_Path path = new GH_Path();
                    GH_Path p = path.AppendElement(i);
                    path = p;

                    resultTree.Add(resultList[i][j], p);
                }
            }

            //Output
            DA.SetDataTree(0, resultTree);
        }
        #endregion
    }   
}



