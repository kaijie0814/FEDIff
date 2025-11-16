/**
 * @file   static_diffusion.h
 * @brief  Main file of the FemFusion program.
 *         A program to solve static neutron diffusion equation with the finite element method.
 */

#ifndef STATIC_DIFFUSION_H
#define STATIC_DIFFUSION_H

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/block_vector.h>

#include <boost/io/ios_state.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <petsc.h>
#include <list>
#include <iomanip>

#include "materials.h"

/**
 *
 */
using namespace dealii;

// Here begins the important class StaticDiffusion that defines all the problem

template <int dim>
  class EnergyGroup
  {
  public:
    EnergyGroup(const unsigned int        group,
                const Materials          &materials,
                const Triangulation<dim> &tria,
                const FiniteElement<dim> &fe);
 
    void setup_linear_system();
 
    unsigned int n_active_cells() const;
    unsigned int n_dofs() const;
 
    void assemble_system_matrix();
    void assemble_ingroup_rhs(const Function<dim> &extraneous_source);
    void assemble_cross_group_rhs(const EnergyGroup<dim> &g_prime);
 
    void solve();
 
    double get_fission_source() const;
 
    void output_results(const unsigned int cycle) const;
 
    void estimate_errors(Vector<float> &error_indicators) const;
 
    void refine_grid(const Vector<float> &error_indicators,
                     const double         refine_threshold,
                     const double         coarsen_threshold);
 
  public:
    Vector<double> solution;
    Vector<double> solution_old;
 
 
  private:
    const unsigned int  group;
    const Materials    &materials;
 
    std::vector<Triangulation<dim>>        triangulation;
    const FiniteElement<dim> &fe;
    DoFHandler<dim>           dof_handler;
 
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
 
    Vector<double> system_rhs;
 
    std::map<types::global_dof_index, double> boundary_values;
    AffineConstraints<double>                 hanging_node_constraints;
 
 
  private:
    void assemble_cross_group_rhs_recursive(
      const EnergyGroup<dim>                        &g_prime,
      const typename DoFHandler<dim>::cell_iterator &cell_g,
      const typename DoFHandler<dim>::cell_iterator &cell_g_prime,
      const FullMatrix<double>                      &prolongation_matrix);    

  };

template <int dim>
  class StaticDiffusion
  {
    public:

      StaticDiffusion (ParameterHandler &prm,
        const bool verbose = false,
        const bool silent = false);

    class Parameters {
      public:
        static void declare_parameters (ParameterHandler &prm);
        // void        get_parameters(ParameterHandler &prm);
      };

    void run (ParameterHandler &prm);

    // Assemble
    void initialize_problem (ParameterHandler &prm);

    void refine_grid();

    double get_total_fission_source() const;

    // Output results
    // void postprocess ();
    // void output_results () const;

    ParameterHandler &prm;

    // Cout streams
    ConditionalOStream verbose_cout;
    ConditionalOStream materials_cout;
    ConditionalOStream cout;

    // Some problem parameters
    std::vector<unsigned int> assem_per_dim;
    std::vector<std::vector<double> > assembly_pitch;
    std::vector<double> power_axial;
    std::vector<double> volume_per_plane;
    static std::vector<unsigned int> boundary_conditions;

    const int n_fe_degree;
    const unsigned int n_groups;
    unsigned int n_refinements;
    unsigned int n_dofs;
    unsigned int n_cells;
    unsigned int n_assemblies;
    unsigned int n_out_ref;

    // added
    double convergence_tolerance;
    std::string tria_file;
    std::string vec_file;
    std::string geom_file;
    std::string xs_file;

    TransportMatrix<dim, n_fe_degree> T;
    FisionMatrix<dim, n_fe_degree> F;

    FE_Q<dim> fe;
    Triangulation<dim> tria;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;

    Timer timer;

    Materials materials;
    

    double k_eff;

    std::vector<std::unique_ptr<EnergyGroup<dim>>> energy_groups;

    // std::vector<double> albedo_factors;

  };

#endif /* STATIC_DIFFUSION_H */

