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
    /// Restricted Boltzmann Node
    /// </summary>
    public class nodeRestrictedBoltzmann : GH_Component
    {
        #region Register Node

        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeRestrictedBoltzmann()
            : base("Restricted Boltzmann Machine", "ResBoltz", "Solver for Restricted Boltzmann machines.", "LunchBox", "Machine Learning")
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
            get { return new Guid("32864682-9c58-4484-b3cf-6d32298295d7"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_RestrictedBolz; }
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
            List<double> test = new List<double>();
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            GH_Structure<GH_Number> outputs = new GH_Structure<GH_Number>();

            //Tree Variables
            DA.GetDataList(0, test);
            DA.GetDataTree<GH_Number>(1, out inputs);
            DA.GetDataTree<GH_Number>(2, out outputs);

            // list of lists
            List<List<double>> inputList = new List<List<double>>();
            List<List<double>> outputList = new List<List<double>>();

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
            double[] result = learning.RestrictedBoltzmann(test, inputList, outputList);

            //Output
            DA.SetDataList(0, result.ToList());

        }
        #endregion
    }  
}



