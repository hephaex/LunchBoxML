using System;
using System.Collections.Generic;
using System.Drawing;

using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Logistic Regression Node
    /// </summary>
    public class nodeLogisticRegression : GH_Component
    {
        #region Register Node

        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeLogisticRegression()
            : base("Logistic Regression", "LogReg", "Solver for Logistic regression problems.", "LunchBox", "Machine Learning")
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
            get { return new Guid("c53baefd-7dde-4be7-b89a-4586da0254e7"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_LogisticRegression; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter ("Test Data", "Test", "Data to test against learning data.", GH_ParamAccess.list);
            pManager.AddNumberParameter("Inputs", "Inputs", "The list of inputs.", GH_ParamAccess.tree);
            pManager.AddBooleanParameter ("Output", "Output", "The list of Outputs.", GH_ParamAccess.list);

        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Resultant prediction");
            pManager.Register_GenericParam("Score", "Score", "The predicted scores");

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
            List<double> test = new List<double>(); 
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            List<bool> outputs = new List<bool>();

            //Tree Variables
            DA.GetDataList(0, test);
            DA.GetDataTree<GH_Number>(1, out inputs);
            DA.GetDataList(2, outputs);

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

            //Result
            clsML learning = new clsML();
            Tuple<bool,double> result = learning.LogisticRegression(test, inputList, outputs);
 
            //Output
            DA.SetData(0, result.Item1);
            DA.SetData(1, result.Item2);
        }
        #endregion
    } 
}


    
