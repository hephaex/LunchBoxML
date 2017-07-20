using System;
using System.Collections.Generic;
using System.Drawing;

using Grasshopper.Kernel;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Linear Regression Node
    /// </summary>
    public class nodeLinearRegression: GH_Component
    {
        #region Register Node

        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeLinearRegression()
            : base("Linear Regression", "LineReg", "Solver for linear regression problems.", "LunchBox", "Machine Learning")
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
            get { return new Guid("97e94daa-7958-489a-aaaa-223c6f76bfc4"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_LinearRegression; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter ("Test Data", "Test", "Data to test against learning data.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Inputs", "Inputs", "The list of inputs.", GH_ParamAccess.list);
            pManager.AddNumberParameter("Output", "Output", "The list of Outputs.", GH_ParamAccess.list);
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
            // Solution

            //Variables
            double test = double.NaN;
            List<double> inputs = new List<double>();
            List<double> outputs = new List<double>();
            
            DA.GetData(0, ref test);
            DA.GetDataList(1, inputs);
            DA.GetDataList(2, outputs);

            //Result
            clsML learning = new clsML();
            double result = learning.LinearRegression(test, inputs, outputs);

            //Output
            DA.SetData(0, result);
        }
        #endregion
    }   
}



