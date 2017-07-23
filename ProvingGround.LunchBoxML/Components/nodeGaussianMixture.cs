using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Data;

using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using ProvingGround.MachineLearning.Classes;

namespace ProvingGround.MachineLearning
{
    /// <summary>
    /// Gaussian Mixture Node
    /// </summary>
    public class nodeGaussianMixture: GH_Component
    {
        #region Register Node
        /// <summary>
        /// Load Node Template
        /// </summary>
        public nodeGaussianMixture()
            : base("Gaussian Mixture", "GaussianMix", "Solver for Gaussian Mixture models.", "LunchBox", "Machine Learning")
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
            get { return new Guid("853076e8-5b04-43cb-b075-c6b469d74da2"); }
        }

        /// <summary>
        /// Icon 24x24
        /// </summary>
        protected override Bitmap Icon
        {
            get { return Properties.Resources.PG_ML_Gaussian; }
        }
        #endregion

        #region Inputs/Outputs
        /// <summary>
        /// Node inputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter("Inputs", "Inputs", "The list of inputs.", GH_ParamAccess.tree);
            pManager.AddIntegerParameter("Components", "Components", "number of clusters", GH_ParamAccess.item, 2);
            pManager.AddIntegerParameter("Random Seed", "Seed", "Randomization seed value.", GH_ParamAccess.item, 5);
        }

        /// <summary>
        /// Node outputs
        /// </summary>
        /// <param name="pManager"></param>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.Register_GenericParam("Result", "Result", "Resultant prediction");
            pManager.Register_GenericParam("Likelihood", "Likelihood", "Log-likelyhood that an input belongs to a cluster.");
            pManager.Register_GenericParam("Probability", "Probability", "Probability that an input belongs to a cluster.");
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
            GH_Structure<GH_Number> inputs = new GH_Structure<GH_Number>();
            int components = 2;
            int seed = 5;

            DA.GetDataTree<GH_Number>(0, out inputs);
            DA.GetData(1, ref components);
            DA.GetData(2, ref seed);

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
            Tuple<int[], double[],double[][]> result = learning.GaussianMixture(inputList,components, seed);
            List<List<double>> resultList3 = result.Item3
                .Where(inner => inner != null) // Cope with uninitialised inner arrays.
                .Select(inner => inner.ToList()) // Project each inner array to a List<string>
                .ToList();

            DataTree<double> result3 = new DataTree<double>();
            for (int i = 0; i < resultList3.Count; i++)
            {
                for (int j = 0; j < resultList3[i].Count; j++)
                {
                    GH_Path path = new GH_Path();
                    GH_Path p = path.AppendElement(i);
                    path = p;

                    result3.Add(resultList3[i][j], p);
                }
            }

            //Output
            DA.SetDataList(0, result.Item1.ToList());
            DA.SetDataList(1, result.Item2.ToList());
            DA.SetDataTree(2, result3);
        }
        #endregion
    }
}



