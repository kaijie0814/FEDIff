/**
 * @file main.cc
 * @brief Main program for the neutron diffusion solver.
 */

#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/petsc_vector.h>

#include "../include/static_diffusion.h"


int main(int argc, char **argv) {
    try 
    {
        using namespace dealii;
        //using namespace femdiffusion;
        deallog.depth_console(0);

        bool verbose = false;
        bool silent = false;

        // Default input file name
        std::string input_filename;
        if (argc < 2)
            input_filename = "project.prm";
        else
            input_filename = argv[1];

        // Default dimension
        // const int dim = 2;

        ParameterHandler prm;

        StaticDiffusion<dim>::Parameters parameters;

        parameters.declare_parameters(prm);

        prm.parse_input(input_filename);

        dim = 

        // Get polynomial degree of the FE
        // const int n_fe_degree = prm.get_integer("FE_Degree");

        StaticDiffusion<dim> StaticDiffusion(prm, verbose, silent);
       
        StaticDiffusion.run(prm);
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}