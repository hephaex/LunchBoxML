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
            pManager.AddIntegerParameter("Hidden Neurons", "Hid Neur", "Number of Hidden Neurons.", GH_ParamAccess.item,5);
            pManager.AddIntegerParameter("Iterations", "Iter", "Number of iterations to teach the network.", GH_ParamAccess.item,10);

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
            GH_Structure<GH_Number> m_tests = new GH_Structure<GH_Number>();
            GH_Structure<GH_Number> m_inputs = new GH_Structure<GH_Number>();
            List<int> m_label = new List<int>();
            int m_numOfNeurons = 5;
            int m_numOfIterations=10;

            //Tree Variables
            DA.GetDataTree<GH_Number>(0, out m_tests);
            DA.GetDataTree<GH_Number>(1, out m_inputs);
            DA.GetDataList(2, m_label);
            DA.GetData(3, ref m_numOfNeurons);
            DA.GetData(4, ref m_numOfIterations);

            // list of lists
            List<List<double>> m_inputList = new List<List<double>>();

            // input list of lists from tree
            for (int i = 0; i < m_inputs.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = m_inputs.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                m_inputList.Add(list);
            }

            List<List<double>> m_testList = new List<List<double>>();

            // input list of lists from tree
            for (int i = 0; i < m_tests.Branches.Count; i++)
            {
                List<double> list = new List<double>(0);
                List<GH_Number> branch = m_tests.Branches[i];
                foreach (GH_Number num in branch)
                {
                    list.Add(num.Value);
                }

                m_testList.Add(list);
            }

            //Result
            clsML learning = new Classes.clsML();
            string[] result = learning.NeuralNetwork(m_testList, m_inputList, m_label,m_numOfNeurons,m_numOfIterations);
 
            //Output
            DA.SetDataList(0, result.ToList());
        }
        #endregion
    }   
}


    
