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
    /// Neural Network Node
    /// </summary>
    public class nodeNeuralNetwork : GH_Component
    {
        #region Register Node

        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeNeuralNetwork()
            : base("Neural Network", "Neural", "Solver for Neural Network problems.", "LunchBox", "Machine Learning")
        {

        }

        /// <summary>
        /// Component Exposure
        /// </summary>
        public override GH_Exposure Exposure
        {
            get { return GH_Exposure.tertiary; }
        }

        /// <summary>
        /// GUID generator http://www.guidgenerator.com/online-guid-generator.aspx
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("75157d8d-81a3-44b5-886c-1e157c42216c"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_NeuralNetowork; }
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
            pManager.AddIntegerParameter ("Labels", "Labels", "The list of Labels.", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Hidden Neurons", "Neurons", "Number of Hidden Neurons.", GH_ParamAccess.item,5);
            pManager.AddNumberParameter("Alpha", "Alpha", "Sigmoid's alpha value.", GH_ParamAccess.tree, 2.0);
            pManager.AddIntegerParameter("Iterations", "Iter", "Number of iterations to teach the network.", GH_ParamAccess.item, 10);
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
            GH_Structure<GH_Number> tests = new GH_Structure<GH_Number>();
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            List<int> labels = new List<int>();
            int numOfNeurons = 5;
            int numOfIterations=10;
            double alpha = 2.0;

            //Tree Variables
            DA.GetDataTree<GH_Number>(0, out tests);
            DA.GetDataTree<GH_Number>(1, out inputs);
            DA.GetDataList(2, labels);
            DA.GetData(3, ref numOfNeurons);
            DA.GetData(4, ref alpha);
            DA.GetData(5, ref numOfIterations);
            
            // list of lists
            List<List<double>> inputList = new List<List<double>>();

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

            List<List<double>> testList = new List<List<double>>();

            // input list of lists from tree
            for (int i = 0; i < tests.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = tests.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                testList.Add(list);
            }

            //Result
            clsML learning = new Classes.clsML();
            string[] result = learning.NeuralNetwork(testList, inputList, labels, numOfNeurons, alpha, numOfIterations);
 
            //Output
            DA.SetDataList(0, result.ToList());
        }
        #endregion
    }   
}


    
