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
    /// Hidden Markov Node
    /// </summary>
    public class nodeHiddenMarkovModel : GH_Component
    {
        #region Register Node

        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeHiddenMarkovModel()
            : base("Hidden Markov Model", "HiddenMark", "Solver for Hidden Markov Model problems.", "LunchBox", "Machine Learning")
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
            get { return new Guid("cfba1813-ae31-4197-994c-1832d3bef2ac"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_HiddenMarkov; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Inputs", "Inputs", "The list of inputs.", GH_ParamAccess.tree);
            pManager.AddIntegerParameter("Generations", "Gen", "Number of new samples to generate", GH_ParamAccess.item, 3);
            pManager.AddIntegerParameter("Random Seed", "Seed", "Random Seed to start generation.", GH_ParamAccess.item, 5);
            pManager.AddIntegerParameter("States", "States", "Number of states to be used in the model.", GH_ParamAccess.item, 4);
        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Predicted generation");

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
            int num = 3;
            int seed = 5;
            int states = 4;
            GH_Structure<GH_String> inputs = new GH_Structure<GH_String>();

            //Tree Variables
            DA.GetDataTree<GH_String>(0, out inputs);
            DA.GetData(1, ref num);
            DA.GetData(2, ref seed);
            DA.GetData(3, ref states);

            // list of lists
            List<List<string>> inputList = new List<List<string>>();

            // input list of lists from tree
            for (int i = 0; i < inputs.Branches.Count; i++)
            {
                List<string> list = new List<string>(0);
                List<GH_String> branch = inputs.Branches[i];
                foreach (GH_String str in branch)
                {
                    list.Add(str.ToString());
                }

                inputList.Add(list);
            }

            //Result
            clsML learning = new clsML();
            string[] result = learning.HiddenMarkovModel(inputList, num, seed, states);

            //Output
            DA.SetDataList(0, result.ToList());

        }
        #endregion
    }
}



