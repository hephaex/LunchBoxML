using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Data;

using Grasshopper.Kernel;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Naive Bayes Node
    /// </summary>
    public class nodeNaiveBayes : GH_Component
    {
        #region Register Node
        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeNaiveBayes()
            : base("Naive Bayes Classification", "NaiveBayes", "Solver for Naive Bayes classification.", "LunchBox", "Machine Learning")
        {

        }

        /// <summary>
        /// Component Exposure
        /// </summary>
        public override GH_Exposure Exposure
        {
            get { return GH_Exposure.secondary; }
        }

        /// <summary>
        /// GUID generator http://www.guidgenerator.com/online-guid-generator.aspx
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("dc1394a2-cacb-492c-8e2f-13b0dd8f26da"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_NaiveBayes; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Test Data", "Test", "Data to test against learning data.", GH_ParamAccess.list);
            pManager.AddGenericParameter("Learning Data", "Data", "Dataset to learn. (LunchBox DataTable)", GH_ParamAccess.item);
            pManager.AddGenericParameter("Inputs", "Inputs", "Names of the input columns.", GH_ParamAccess.list);
            pManager.AddGenericParameter("Output", "Output", "Name of the output column.", GH_ParamAccess.item);
        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Resultant prediction");
            pManager.Register_GenericParam("Num", "Numeric", "The numeric output that represents the answer");
            pManager.Register_GenericParam("Probs", "Probabilities", "Probabilities for each possible answer");
        }
        #endregion

        #region Solution
        /// <summary>
        /// Code by the component
        /// </summary>
        /// <param name="DA"></param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // Solution

            //Variables
            Object data = null;
            List<string> test = new List<string>();
            List<string> inputCols = new List<string>();
            string output = null;

            DA.GetDataList(0, test);
            DA.GetData(1, ref data); 
            DA.GetDataList(2, inputCols);
            DA.GetData(3, ref output);

            //DataTable conversion
            Grasshopper.Kernel.Types.GH_Goo<Object> dtObj = (Grasshopper.Kernel.Types.GH_Goo<Object>)data;
            DataTable m_dt = (DataTable)dtObj.Value;

            //Result
            clsML learning = new clsML();
            Tuple<string, int, double[]> result = learning.NaiveBayesClassifier(test, m_dt, inputCols, output);

            //Output
            DA.SetData(0, result.Item1);
            DA.SetData(1, result.Item2);
            DA.SetDataList(2, result.Item3.ToList());
        }
        #endregion
    }
}



