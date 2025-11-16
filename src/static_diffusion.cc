/**
 * @file   static_diffusion.cc
 * @brief  Implementation of the class StaticDiffusion and the main functions of
 *  the FEMFFUSION program.
 */

#include <deal.II/lac/solver_selector.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/utilities.h>
#include <deal.II/fe/mapping_q.h>
#include <filesystem>

#include "../include/static_diffusion.h"
//#include "../include/prob_geom.h"
#include "../include/input_geom.h"
#include "../include/materials.h"
#include "../include/utils.h"
#include "../include/create_geom.h"
//#include "../include/printing.h"

#include <petscsys.h>

#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

using namespace dealii;
namespace fs = std::filesystem;

template <int dim>
void StaticDiffusion<dim>::Parameters::declare_parameters (ParameterHandler &prm) {
  prm.declare_entry("FE_Degree", "2", Patterns::Integer(1), 
    "Polynomial degree of the finite element to be used");  
  prm.declare_entry("Energy_Groups", "7", Patterns::Integer(1),
    "Number of energy groups");
  prm.declare_entry("N_Refinements", "3", Patterns::Integer(0),
    "Number of adaptive mesh refinement");
  prm.declare_entry("Refinement_Threshold", "0.3", Patterns::Double(0,1),
    "Refinement threshold for mesh refinement (0.0 ~ 1.0)");
  prm.declare_entry("Solver_Convergence", "1e-9", Patterns::Double(0),
    "Convergence tolerance for the linear solver"); 
  prm.declare_entry("Convergence_Tolerance", "1e-7", Patterns::Double(0),
    "Convergence tolerance for power iteration");
  prm.declare_entry("Geometry_Filename", "", Patterns::FileName(),
    "Input geometry file in XML format");
  prm.declare_entry("Triangulation_Filename", "", Patterns::FileName(),
    "Input triangulation file in TRI format");
  prm.declare_entry("Material_Filename", "", Patterns::FileName(),
    "Input material file in XML format");
  // prm.declare_entry("Boundary_Conditions", "3", Patterns::Anything(),
  //   "(LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK ) \n"
  //     "(0  ZeroFlow) (1  Symmetry) (2 Albedo) (3 Vacuum)");
  prm.declare_entry("Global_Refinements", "0", Patterns::Integer(0),
    "Number of global refinements");
  prm.declare_entry("Only_Output","false",Patterns::Bool(),
    "Not executing calculation but only display output");
  prm.declare_entry("Refinement_Mode","adaptive",Patterns::Selection("global|adaptive|energy_dependent|energy_dependent_multilevel"),
    "Refinement mode (global/ adaptive/ energy_dependent)");
  prm.declare_entry("Knudsen_Number","",Patterns::Anything(),
    "Energy dependent mesh indicator");
  prm.declare_entry("Error_Mode", "fine", Patterns::Selection("fine|coarse"),
    "Error mode (fine/ coarse)");
  prm.declare_entry("Fine_Solution_Input","",Patterns::FileName(),
    "Input file for fine solution");
  prm.declare_entry("Fine_Grid_Input", "",Patterns::FileName(),
    "Input file for fine solution grid");
}

/**
 * @brief Constructor of the main class StaticDiffusion.
 * Reads the input file and it reads or builds the grid.
 */
template <int dim>
  StaticDiffusion<dim>::StaticDiffusion (
    ParameterHandler &prm,
    const bool verbose,
    const bool silent) :
      prm(prm),
      verbose_cout(std::cout, verbose),
      materials_cout(std::cout, false), // No verbose for materials
      cout(std::cout, !silent),
      n_fe_degree(prm.get_integer("FE_Degree")),
      n_groups(prm.get_integer("Energy_Groups")),
      n_refinements(prm.get_integer("N_Refinements")),
      refinement_mode(prm.get("Refinement_Mode")),
      refinement_threshold(prm.get_double("Refinement_Threshold")),
      solver_convergence(prm.get_double("Solver_Convergence")),
      convergence_tolerance(prm.get_double("Convergence_Tolerance")),
      global_refinements(prm.get_integer("Global_Refinements")),
      only_output(prm.get_bool("Only_Output")),
      // knudsen_number(prm.get_double("Knudsen_Number")),
      error_mode(prm.get("Error_Mode")),
      fine_solution_input(prm.get("Fine_Solution_Input")),
      fine_grid_input(prm.get("Fine_Grid_Input")),
      fe(n_fe_degree),
      dof_handler(tria),
      materials(materials_cout),
      timer_output(std::cout, TimerOutput::never, TimerOutput::wall_times)
  { 
    parse_vector(prm.get("Knudsen_Number"), knudsen_number);
  }

template <int dim>
  void StaticDiffusion<dim>::initialize_problem (ParameterHandler &prm)
  {
    TimerOutput::Scope scope(timer_output, "Initialization and setup geometry");

    verbose_cout << "Start of the program " << std::endl;
    AssertRelease(n_fe_degree > 0, "FE can not be 0");

    // ---------------------------------------------------------------------------------
    // GEOMETRY SHAPE
    bool listen_to_material_id = true;

    // ---------------------------------------------------------------------------- //
    // GEOMETRY
    // Mesh File

    // (geo_type == "Composed")

    tria_file = prm.get("Triangulation_Filename");
    vec_file = tria_file + ".vec";

    geom_file = prm.get("Geometry_Filename");
    verbose_cout << "geom_file " << geom_file << std::endl;

    verbose_cout << "load geometry file... " << std::flush;
    InputGeom input_geometry;
    input_geometry.load(geom_file);
    verbose_cout << "Done!" << std::endl;

    xs_file = prm.get("Material_Filename");
    verbose_cout << "xs_file " << xs_file << std::endl;

    // Fill assem_per_dim
    assem_per_dim.resize(3, 1);
    for (unsigned int d = 0; d < dim; ++d)
    {
      assem_per_dim[d] = input_geometry.core.n_planes;
    }
    n_assemblies = assem_per_dim[0] * assem_per_dim[1] * assem_per_dim[2];

    materials.reinit(xs_file, n_groups, assem_per_dim,
      n_assemblies, listen_to_material_id);

    
    // Load Triangulation
    if (fexists(tria_file))
    {
      verbose_cout << "load_triangualtion... " << std::flush;
      GridIn<dim> gridin;
      std::ifstream f(tria_file.c_str());
      gridin.attach_triangulation(tria);
      gridin.read_ucd(f);

      std::vector<unsigned int> user_indices;
      parse_vector_in_file(vec_file, "User_indices", user_indices,
        tria.n_active_cells(), tria.n_active_cells());

      unsigned int i = 0;
      typename Triangulation<dim>::active_cell_iterator cell =
                                                                tria.begin_active();
      for (; cell != tria.end(); ++cell, ++i)
      {
        cell->set_user_index(user_indices[i]);
      }

      verbose_cout << "Done!" << std::endl;
    }
    else
    {
      make_grid(input_geometry, tria, materials);
    }



    // Make global refinement
    tria.refine_global(global_refinements);

    // if (error_mode == "fine")
    // {
    //   GridOut fine_grid;
    //   std::ofstream out("fine_grid.msh");
    //   grid_out.write_msh(tria, out);
    // }

    // Set Geometry Matrix
    verbose_cout << "  reading geometry_matrix... " << std::flush;
    std::vector<unsigned int> geo_ps = default_geometry_points(
      assem_per_dim);
    materials.set_geometry_matrix(assem_per_dim, geo_ps);
    verbose_cout << " Done!" << std::endl;

    // Copy Boundary Conditions
    boundary_conditions = input_geometry.core.boundary[0];
    for (unsigned int d = 1; d < dim; ++d)
      boundary_conditions.insert(boundary_conditions.end(),
        input_geometry.core.boundary[0].begin(),
        input_geometry.core.boundary[0].end());

    for (unsigned int group = 0; group < n_groups; ++group)
      energy_groups.emplace_back(std::make_unique<EnergyGroup<dim>>(
        group, materials, tria, fe, timer_output));
  }


template <int dim>
  EnergyGroup<dim>::EnergyGroup(const unsigned int        group,
                                const Materials          &materials,
                                const Triangulation<dim> &tria,
                                const FiniteElement<dim> &fe,
                                TimerOutput        &timer_output)
    : group(group)
    , mapping(3)
    , materials(materials)
    , fe(fe)
    , dof_handler(triangulation)
    , timer_output(timer_output)
  {
    triangulation.copy_triangulation(tria);
    dof_handler.distribute_dofs(fe);
  }  

template <int dim>
  unsigned int EnergyGroup<dim>::n_active_cells() const
  {
    return triangulation.n_active_cells();
  }
 
template <int dim>
  unsigned int EnergyGroup<dim>::n_dofs() const
  {
    return dof_handler.n_dofs();
  }
 
template <int dim>
std::vector<unsigned int> StaticDiffusion<dim>::boundary_conditions; 
 
template <int dim>
  void EnergyGroup<dim>::setup_linear_system()
  {

    const unsigned int n_dofs = dof_handler.n_dofs();
 
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();
 
    system_matrix.clear();
 
    DynamicSparsityPattern dsp(n_dofs, n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    hanging_node_constraints.condense(dsp);
    sparsity_pattern.copy_from(dsp);
 
    system_matrix.reinit(sparsity_pattern);
    // matrix_L.reinit(sparsity_pattern);
    // matrix_F.reinit(sparsity_pattern);
 
    system_rhs.reinit(n_dofs);
 
    if (solution.empty())
      {
        solution.reinit(n_dofs);
        solution_old.reinit(n_dofs);
        solution_old = 1.0;
        solution     = solution_old;
      }
 
    boundary_values.clear();

    for (unsigned int c = 0; c < StaticDiffusion<dim>::boundary_conditions.size(); c++)
      if (StaticDiffusion<dim>::boundary_conditions[c] == 2)
      {
        // DoFTools::make_zero_boundary_constraints(dof_handler, c,
        //   hanging_node_constraints);
        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 c,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);
      }
  }

template <int dim>
  void EnergyGroup<dim>::assemble_system_matrix()
  {

    const QGauss<dim> quadrature_formula(fe.degree + 1);
 
    FEValues<dim> fe_values(mapping, fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_JxW_values);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
 
        fe_values.reinit(cell);
 
        const double diffusion_coefficient =
          materials.get_diffusion_coefficient(group, cell->material_id());
        const double removal_XS =
          materials.get_sigma_r(group, cell->material_id());
 
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                ((diffusion_coefficient * fe_values.shape_grad(i, q_point) *
                    fe_values.shape_grad(j, q_point) +
                  removal_XS * fe_values.shape_value(i, q_point) *
                    fe_values.shape_value(j, q_point)) *
                 fe_values.JxW(q_point));
 
        cell->get_dof_indices(local_dof_indices);
 
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
      }
 
    hanging_node_constraints.condense(system_matrix);

    // matrix_L.copy_from(system_matrix);
    // hanging_node_constraints.condense(matrix_L);
  }
 
 
  template <int dim>
  void
  EnergyGroup<dim>::assemble_ingroup_rhs(const Function<dim> &extraneous_source)
  {

    system_rhs.reinit(dof_handler.n_dofs());

    const QGauss<dim> quadrature_formula(fe.degree + 1);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    FEValues<dim> fe_values(mapping, fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);
 
    Vector<double>      cell_rhs(dofs_per_cell);
    std::vector<double> extraneous_source_values(n_q_points);
    std::vector<double> solution_old_values(n_q_points);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;
 
        fe_values.reinit(cell);
 
        const double fission_dist_XS =
          materials.get_xi_nu_sigma_f(group, group, cell->material_id());
 
        extraneous_source.value_list(fe_values.get_quadrature_points(),
                                     extraneous_source_values);
 
        fe_values.get_function_values(solution_old, solution_old_values);
 
        cell->get_dof_indices(local_dof_indices);
 
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) +=
              ((extraneous_source_values[q_point] +
                fission_dist_XS * solution_old_values[q_point]) *
               fe_values.shape_value(i, q_point) * fe_values.JxW(q_point));
 
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
  }
 
 
 
  template <int dim>
  void
  EnergyGroup<dim>::assemble_cross_group_rhs(const EnergyGroup<dim> &g_prime)
  {
    if (group == g_prime.group)
      return;
 
    const std::list<std::pair<typename DoFHandler<dim>::cell_iterator,
                              typename DoFHandler<dim>::cell_iterator>>
      cell_list =
        GridTools::get_finest_common_cells(dof_handler, g_prime.dof_handler);

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(mapping, fe,
                            quadrature_formula,
                            update_values | update_JxW_values);
 
    for (const auto &cell_pair : cell_list)
      {
        FullMatrix<double> unit_matrix(fe.n_dofs_per_cell());
        for (unsigned int i = 0; i < unit_matrix.m(); ++i)
          unit_matrix(i, i) = 1;
        assemble_cross_group_rhs_recursive(g_prime,
                                           cell_pair.first,
                                           cell_pair.second,
                                           unit_matrix,
                                           fe_values);
      }
  }
 
 
 
  template <int dim>
  void EnergyGroup<dim>::assemble_cross_group_rhs_recursive(
    const EnergyGroup<dim>                        &g_prime,
    const typename DoFHandler<dim>::cell_iterator &cell_g,
    const typename DoFHandler<dim>::cell_iterator &cell_g_prime,
    const FullMatrix<double>                      &prolongation_matrix,
    FEValues<dim>                                 &fe_values)
  {

    if (!cell_g->has_children() && !cell_g_prime->has_children())
      {
        TimerOutput::Scope scope(timer_output,"Assemble local mass matrix");
        const QGauss<dim>  quadrature_formula(fe.degree + 1);
        const unsigned int n_q_points = quadrature_formula.size();
 
        // FEValues<dim> fe_values(mapping, fe,
        //                         quadrature_formula,
        //                         update_values | update_JxW_values);
 
        if (cell_g->level() > cell_g_prime->level())
          fe_values.reinit(cell_g);
        else
          fe_values.reinit(cell_g_prime);
 
        const double fission_dist_XS =
          materials.get_xi_nu_sigma_f(g_prime.group,  // nu_sigma_f [from_group]
                                      group,          // chi[to_group]
                                      cell_g_prime->material_id());
 
        const double scattering_XS =
          materials.get_sigma_s(g_prime.group, // from_group
                                    group,     // to_group
                                    cell_g_prime->material_id());
 
        FullMatrix<double> local_mass_matrix_f(fe.n_dofs_per_cell(),
                                               fe.n_dofs_per_cell()); // fission source
        FullMatrix<double> local_mass_matrix_g(fe.n_dofs_per_cell(),
                                               fe.n_dofs_per_cell()); // scattering source
 
        {
          TimerOutput::Scope scope(timer_output, "Get local mass matrix");
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              {
                local_mass_matrix_f(i, j) +=
                  (fission_dist_XS * fe_values.shape_value(i, q_point) *
                   fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));
                local_mass_matrix_g(i, j) +=
                  (scattering_XS * fe_values.shape_value(i, q_point) *
                   fe_values.shape_value(j, q_point) * fe_values.JxW(q_point));
              }
        }
 
        Vector<double> g_prime_new_values(fe.n_dofs_per_cell());
        Vector<double> g_prime_old_values(fe.n_dofs_per_cell());
        cell_g_prime->get_dof_values(g_prime.solution_old, g_prime_old_values);
        cell_g_prime->get_dof_values(g_prime.solution, g_prime_new_values);
 
        Vector<double> cell_rhs(fe.n_dofs_per_cell());
        Vector<double> tmp(fe.n_dofs_per_cell());
 
        {
          TimerOutput::Scope scope(timer_output, "Calculate prolongation matrix");
          if (cell_g->level() > cell_g_prime->level())
          {
            prolongation_matrix.vmult(tmp, g_prime_old_values);
            local_mass_matrix_f.vmult(cell_rhs, tmp);
 
            prolongation_matrix.vmult(tmp, g_prime_new_values);
            local_mass_matrix_g.vmult_add(cell_rhs, tmp);
          }
          else
          {
            local_mass_matrix_f.vmult(tmp, g_prime_old_values);
            prolongation_matrix.Tvmult(cell_rhs, tmp);
 
            local_mass_matrix_g.vmult(tmp, g_prime_new_values);
            prolongation_matrix.Tvmult_add(cell_rhs, tmp);
          }
        }
 
        std::vector<types::global_dof_index> local_dof_indices(
          fe.n_dofs_per_cell());
        cell_g->get_dof_indices(local_dof_indices);
 
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          system_rhs(local_dof_indices[i]) += cell_rhs(i);

        // Assemble operator
        // for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        //   for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        //       {
        //         matrix_F.add(local_dof_indices[i],
        //                       local_dof_indices[j],
        //                       local_mass_matrix_f(i, j));
        //         matrix_L.add(local_dof_indices[i],
        //                       local_dof_indices[j],
        //                       -local_mass_matrix_g(i, j));
        //       }
        // hanging_node_constraints.condense(matrix_L);
        // hanging_node_constraints.condense(matrix_F);
      }
 
    else
      for (unsigned int child = 0;
           child < GeometryInfo<dim>::max_children_per_cell;
           ++child)
        {
          //TimerOutput::Scope scope(timer_output, "Assemble cross group RHS recursive");
          FullMatrix<double> new_matrix(fe.n_dofs_per_cell(),
                                        fe.n_dofs_per_cell());
          fe.get_prolongation_matrix(child).mmult(new_matrix,
                                                  prolongation_matrix);
 
          if (cell_g->has_children())
            assemble_cross_group_rhs_recursive(g_prime,
                                               cell_g->child(child),
                                               cell_g_prime,
                                               new_matrix,
                                               fe_values);
          else
            assemble_cross_group_rhs_recursive(g_prime,
                                               cell_g,
                                               cell_g_prime->child(child),
                                               new_matrix,
                                               fe_values);
        }
  }
 
 
  template <int dim>
  double EnergyGroup<dim>::get_fission_source() const
  {
    const QGauss<dim>  quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
 
    FEValues<dim> fe_values(mapping, fe,
                            quadrature_formula,
                            update_values | update_JxW_values);
 
    std::vector<double> solution_values(n_q_points);
 
    double fission_source = 0;
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
 
        const double fission_XS =
          materials.get_nu_sigma_f(group, cell->material_id());
 
        fe_values.get_function_values(solution, solution_values);
 
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) 
          fission_source +=
            (fission_XS * solution_values[q_point] * fe_values.JxW(q_point));
      }
 
    return fission_source;
  }
 
 
  template <int dim>
  int EnergyGroup<dim>::solve(const double solver_convergence)
  {

    hanging_node_constraints.condense(system_rhs);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
 
    SolverControl            solver_control(system_matrix.m(),
                                 solver_convergence * system_rhs.l2_norm(),true);
    
    SolverCG<Vector<double>> cg(solver_control);
 
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);
 
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
 
    hanging_node_constraints.distribute(solution);

    return solver_control.last_step();
  }

// template <int dim>
//   void EnergyGroup<dim>::vmult_L(Vector<double> &out, const Vector<double> &in) const
//   {
//     matrix_L.vmult(out, in);
//   }
 
// template <int dim>
//   void EnergyGroup<dim>::vmult_F(Vector<double> &out, const Vector<double> &in) const
//   {
//     matrix_F.vmult(out, in);
//   } 
 
  template <int dim>
  void EnergyGroup<dim>::estimate_errors(Vector<float> &error_indicators) const
  {
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      error_indicators);
    error_indicators /= solution.linfty_norm();
  }
 
 
 
 template <int dim>
  void EnergyGroup<dim>::refine_grid(const Vector<float> &error_indicators,
                                     const double         refine_threshold,
                                     const double         coarsen_threshold)
  {

    for (const auto &cell : triangulation.active_cell_iterators())
      if (error_indicators(cell->active_cell_index()) > refine_threshold)
        cell->set_refine_flag();
      else if (error_indicators(cell->active_cell_index()) < coarsen_threshold)
        cell->set_coarsen_flag();
 
    SolutionTransfer<dim> soltrans(dof_handler);
 
    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement(solution);
 
    triangulation.execute_coarsening_and_refinement();

    dof_handler.distribute_dofs(fe);
    setup_linear_system();
 
    solution.reinit(dof_handler.n_dofs());
    soltrans.interpolate(solution);
 
    hanging_node_constraints.distribute(solution);
 
    solution_old.reinit(dof_handler.n_dofs());
    solution_old = solution;
  }

 template <int dim>
  void EnergyGroup<dim>::refine_global_grid()
  {
    SolutionTransfer<dim> soltrans(dof_handler);
 
    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement(solution);
 
    triangulation.refine_global(1);

    dof_handler.distribute_dofs(fe);
    setup_linear_system();
 
    solution.reinit(dof_handler.n_dofs());
    soltrans.interpolate(solution);
 
    hanging_node_constraints.distribute(solution);
 
    solution_old.reinit(dof_handler.n_dofs());
    solution_old = solution;
  }

/**
 * @brief Refine grid for energy dependent mode (once-time)
 */
 template <int dim>
  void EnergyGroup<dim>::refine_energy_grid (unsigned int group, std::vector<double> kn)
  {

    bool refinement_needed = true;
    unsigned int max_refinement = 10;
    unsigned int count = 0;

    while (refinement_needed && count < max_refinement)
    {
      refinement_needed = false;
      for (const auto &cell : triangulation.active_cell_iterators())
      {
        // double chord_length = chord_length_per_material[cell->material_id()];
        const double total_XS = materials.get_sigma_t(group, cell->material_id());
        const double chord_length = 1.0 / (total_XS * kn[cell->material_id()]);
        if (cell->diameter() > chord_length)
        {
          cell->set_refine_flag();
          refinement_needed = true;
        }
      }

      if (refinement_needed)
        triangulation.execute_coarsening_and_refinement();

      count++;
    }

    dof_handler.distribute_dofs(fe);
    // setup_linear_system();

    // solution.reinit(dof_handler.n_dofs());
    // solution_old.reinit(dof_handler.n_dofs());
    // solution_old = 1.0;
    // solution     = solution_old;
  }

 template <int dim>
  void EnergyGroup<dim>::refine_energy_grid_multilevel (unsigned int group, std::vector<double> kn)
  {
    bool refinement_needed = false;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      // double chord_length = chord_length_per_material[cell->material_id()];
      const double total_XS = materials.get_sigma_t(group, cell->material_id());
      const double chord_length = 1.0 / (total_XS * kn[cell->material_id()]);
      if (cell->diameter() > chord_length)
      {
        cell->set_refine_flag();
        refinement_needed = true;
      }
    }

    if (refinement_needed)
      triangulation.execute_coarsening_and_refinement();

    SolutionTransfer<dim> soltrans(dof_handler);
 
    triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement(solution);
 
    triangulation.execute_coarsening_and_refinement();

    dof_handler.distribute_dofs(fe);
    setup_linear_system();
 
    solution.reinit(dof_handler.n_dofs());
    soltrans.interpolate(solution);
 
    hanging_node_constraints.distribute(solution);
 
    solution_old.reinit(dof_handler.n_dofs());
    solution_old = solution;
  }

 template <int dim>
  int EnergyGroup<dim>::count_energy_grid_refinement (unsigned int group, std::vector<double> kn)
  {

    bool refinement_needed = true;
    int max_refinement = 10;
    int count = 0;

    while (refinement_needed && count < max_refinement)
    {
      refinement_needed = false;
      for (const auto &cell : triangulation.active_cell_iterators())
      {
        // double chord_length = chord_length_per_material[cell->material_id()];
        const double total_XS = materials.get_sigma_t(group, cell->material_id());
        const double chord_length = 1.0 / (total_XS * kn[cell->material_id()]);
        if (cell->diameter() > chord_length)
        {
          cell->set_refine_flag();
          refinement_needed = true;
        }
      }

      if (refinement_needed)
        triangulation.execute_coarsening_and_refinement();

      count++;
    }

    return count;
  }

  template <int dim>
  void EnergyGroup<dim>::output_results(const unsigned int cycle) const
  {

    const std::string grid_folder = "output/grid/";
    const std::string vtk_folder = "output/vtk/";
    const std::string tri_folder = "output/tri/";

    fs::create_directories(grid_folder);
    fs::create_directories(vtk_folder);
    fs::create_directories(tri_folder);

    unsigned int n_group = group + 1;

    //Grid output in svg file
    GridOut grid_out;
    const std::string grid_filename = grid_folder + "grid-FE" +
                                 Utilities::int_to_string(fe.degree, 1) + "." +
                                 Utilities::int_to_string(n_group, 2) + "." +
                                 Utilities::int_to_string(cycle, 2) + ".svg";
    std::ofstream out(grid_filename);
    grid_out.write_svg(triangulation, out);

    // vtk data output
    DataOut<dim> data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "flux");
    // Add material_id as cell data
    Vector<double> material_ids(triangulation.n_active_cells());
    const int output_mesh_order = 4;

    unsigned int i = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        material_ids[i] = cell->material_id();
        ++i;
      }
    data_out.add_data_vector(material_ids, "material_id");
    data_out.build_patches(mapping, output_mesh_order, DataOut<dim>::curved_inner_cells);
    std::string vtk_filename = vtk_folder + "solution-FE" +
                               Utilities::int_to_string(fe.degree, 1) + "." +
                               Utilities::int_to_string(n_group, 2) + "." +
                               Utilities::int_to_string(cycle, 2) + ".vtk";
    std::ofstream vtk_output(vtk_filename);
    data_out.write_vtk(vtk_output);


    // Triangulation outpuut
    std::string tri_filename = tri_folder + "tri-FE" +
                               Utilities::int_to_string(fe.degree, 1) + "." +
                               Utilities::int_to_string(n_group, 2) + "." +
                               Utilities::int_to_string(cycle, 2) + ".tri";
    std::ofstream tri_output(tri_filename);
    GridOutFlags::Ucd gridout_flags(true, true, true);
    grid_out.set_flags(gridout_flags);
    grid_out.write_ucd<dim>(triangulation, tri_output);
  }

template <int dim>
 double EnergyGroup<dim>::compute_l2_norm() 
 {
  Vector<double> norm_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(mapping, dof_handler,
                                    solution,
                                    Functions::ZeroFunction<dim>(),
                                    norm_per_cell,
                                    QGauss<dim>(fe.degree + 1),
                                    VectorTools::L2_norm);
  return VectorTools::compute_global_error(triangulation,
                                    norm_per_cell,
                                    VectorTools::L2_norm);
 }

template <int dim>
 double EnergyGroup<dim>::compute_linfty_norm() 
 {
  const QTrapezoid<1>  q_trapez;
  const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
  Vector<double> norm_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(mapping, dof_handler,
                                    solution,
                                    Functions::ZeroFunction<dim>(),
                                    norm_per_cell,
                                    q_iterated,
                                    VectorTools::Linfty_norm);
  return VectorTools::compute_global_error(triangulation,
                                    norm_per_cell,
                                    VectorTools::Linfty_norm);
 }

// template <int dim>
// void EnergyGroup<dim>::compute_error_norm()
// {

// }

template <int dim>
  double StaticDiffusion<dim>::get_total_fission_source() const
  {
    std::vector<double>  fission_sources(n_groups),flux(n_groups);
    Threads::TaskGroup<> tasks;
    for (unsigned int group = 0; group < n_groups; ++group)
      tasks += Threads::new_task<>([&, group]() {
        fission_sources[group] = energy_groups[group]->get_fission_source();
        flux[group] = energy_groups[group]->solution.l1_norm();
      });
    tasks.join_all();
 
    return std::accumulate(fission_sources.begin(), fission_sources.end(), 0.0);
  }

template <int dim>
  void StaticDiffusion<dim>::refine_grid ()
  {
    std::vector<types::global_dof_index> n_cells(n_groups);
    for (unsigned int group = 0; group < n_groups; ++group)
      n_cells[group] = energy_groups[group]->n_active_cells();
 
    BlockVector<float> group_error_indicators(n_cells);
 
    {
      // Threads::TaskGroup<> tasks;
      for (unsigned int group = 0; group < n_groups; ++group)
        // tasks += Threads::new_task([&, group]() 
        // {
          energy_groups[group]->estimate_errors(
            group_error_indicators.block(group));
        // });
    }
 
    const float max_error         = group_error_indicators.linfty_norm();
    const float refine_threshold  = refinement_threshold * max_error;
    const float coarsen_threshold = 0.01 * max_error;
 
    {
      Threads::TaskGroup<void> tasks;
      for (unsigned int group = 0; group < n_groups; ++group)
        tasks += Threads::new_task([&, group]() {
          energy_groups[group]->refine_grid(group_error_indicators.block(group),
                                            refine_threshold,
                                            coarsen_threshold);
        });
    }
  }

  template <int dim>
  void StaticDiffusion<dim>::refine_global_grid ()
  {
    Threads::TaskGroup<> tasks;
    for (unsigned int group = 0; group < n_groups; ++group)
      tasks += Threads::new_task([&, group]() {
        energy_groups[group]->refine_global_grid();
      });
  }

template <int dim>
  void StaticDiffusion<dim>::refine_energy_grid (std::vector<double> knudsen_number)
  {
    Threads::TaskGroup<> tasks;
    for (unsigned int group = 0; group < n_groups; ++group)
      tasks += Threads::new_task([&, group]() {
        energy_groups[group]->refine_energy_grid(group, knudsen_number);
      });
  }

template <int dim>
  void StaticDiffusion<dim>::refine_energy_grid_multilevel (std::vector<double> knudsen_number)
  {
    Threads::TaskGroup<> tasks;
    for (unsigned int group = 0; group < n_groups; ++group)
      tasks += Threads::new_task([&, group]() {
        energy_groups[group]->refine_energy_grid_multilevel(group, knudsen_number);
      });
  }

template <int dim>
  int StaticDiffusion<dim>::count_energy_grid_refinement (std::vector<double> knudsen_number)
  {
    Vector<int> energy_grid_refinement(n_groups);
    for (unsigned int group = 0; group < n_groups; ++group)
        energy_grid_refinement[group] = energy_groups[group]->count_energy_grid_refinement(group, knudsen_number);
    unsigned int n_refinements = *std::max_element(energy_grid_refinement.begin(), energy_grid_refinement.end());
    return n_refinements;
  }


  /**
 * @brief This is the function which has the top-level control over
 * everything. It also prints some results and time-line.
 */
template <int dim>
  void StaticDiffusion<dim>::run (ParameterHandler &prm)
  {
    timer.start();
    PetscLogDouble memory;

    boost::io::ios_flags_saver restore_flags(std::cout);
    std::cout << std::setprecision(8) << std::fixed;

    double k_eff_old = 1.0;

    for (unsigned int cycle = 0; cycle < n_refinements; cycle++)
    {
      if (cycle == 0)
        {
          initialize_problem(prm);

          if (refinement_mode == "energy_dependent")
          {
            TimerOutput::Scope scope(timer_output, "Refine grid (energy dependent)");
            cout << "   Energy-based refinement" << std::endl;
            refine_energy_grid(knudsen_number);
            n_refinements = 0; // exit after energy refinement
          }
          else if (refinement_mode == "energy_dependent_multilevel")
            n_refinements = count_energy_grid_refinement(knudsen_number);

          {
            TimerOutput::Scope scope(timer_output, "Setup linear system");
            for (unsigned int group = 0; group < n_groups; ++group)
              energy_groups[group]->setup_linear_system();
          }

          cout << "   Total Energy Groups:  " << n_groups << std::endl;

          
          cout << "   Numbers of active cells:       ";
          for (unsigned int group = 0; group < n_groups; ++group)
            cout << energy_groups[group]->n_active_cells() << ' ';
          cout << std::endl;
          
          cout << "   Numbers of degrees of freedom: ";
          for (unsigned int group = 0; group < n_groups; ++group)
            cout << energy_groups[group]->n_dofs() << ' ';
          cout << std::endl;

          cout << "   Refinement mode: " << refinement_mode << std::endl;
          
          cout << "Refinement cycle " << cycle << ":" << std::endl;

          if (only_output)
          {
            for (unsigned int group = 0; group < n_groups; ++group)
              energy_groups[group]->output_results(cycle);
            cout << "Time: " << timer.cpu_time() << " s" << std::endl;
            return;            
          }

        }
      else
        {
          if (refinement_mode == "global")
          {
            TimerOutput::Scope(timer_output, "Refine grid (global)");
            refine_global_grid();
          }
          else if (refinement_mode == "adaptive")
          {
            TimerOutput::Scope scope(timer_output, "Refine grid (adaptive)");
            refine_grid();
          }
          else if (refinement_mode == "energy_dependent")
          {
            TimerOutput::Scope scope(timer_output, "Refine grid (energy dependent multilevel)");
            refine_energy_grid_multilevel(knudsen_number);
          }
          
          cout << "   Numbers of active cells:       ";
          for (unsigned int group = 0; group < n_groups; ++group)
            cout << energy_groups[group]->n_active_cells() << ' ';
          cout << std::endl;
          
          cout << "   Numbers of degrees of freedom: ";
          for (unsigned int group = 0; group < n_groups; ++group)
            cout << energy_groups[group]->n_dofs() << ' ';
          cout << std::endl << std::endl;

          cout << "Refinement cycle " << cycle << ":" << std::endl;
          for (unsigned int group = 0; group < n_groups; ++group)
            energy_groups[group]->solution *= k_eff;
        }

      {
        TimerOutput::Scope scope(timer_output, "Assemble system matrix");
        Threads::TaskGroup<> tasks;
        for (unsigned int group = 0; group < n_groups; ++group)
          tasks += Threads::new_task(
            [&, group]() { energy_groups[group]->assemble_system_matrix(); });
        tasks.join_all();
      }

      double       error;
      unsigned int iteration = 1;
      int inner_iteration = 0;
      do
        {
          {
            TimerOutput::Scope scope(timer_output, "Assemble ingroup RHS");
            for (unsigned int group = 0; group < n_groups; ++group)
            {
              energy_groups[group]->assemble_ingroup_rhs(
                Functions::ZeroFunction<dim>());

              {
                TimerOutput::Scope scope(timer_output, "Assemble cross group RHS");
                for (unsigned int bgroup = 0; bgroup < n_groups;
                    ++bgroup)
                energy_groups[group]->assemble_cross_group_rhs(
                  *energy_groups[bgroup]);
              }

              {
                TimerOutput::Scope scope(timer_output, "Solve linear system");
                inner_iteration += energy_groups[group]->solve(solver_convergence);
              }
            }
          }

          // Compute rayleigh quotient
          // double numerator = 0.0;
          // double denominator = 0.0;
          // for (unsigned int group = 0; group < n_groups; ++group)
          // {
          //   const Vector<double> &phi = energy_groups[group]->solution;
          //   Vector<double> wF(phi.size()), wL(phi.size());

          //   energy_groups[group]->vmult_F(wF, phi);
          //   energy_groups[group]->vmult_L(wL, phi);

          //   numerator   += phi * wF;
          //   denominator += phi * wL;
          // }

          // double rq_factor = numerator / denominator;

          k_eff = get_total_fission_source();

          // double k_new = k_eff / (1.0 + k_eff * rq_factor);

          // k_eff = 0.5 * k_eff + 0.5 * k_new;

          error = std::abs(k_eff - k_eff_old) / std::abs(k_eff);

          std::cout << "Iter number:" << std::setw(3) << std::right
                    << iteration 
                    << " k_eff = " << k_eff
                    << std::endl;
          k_eff_old = k_eff;

          for (unsigned int group = 0; group < n_groups; ++group)
            {
              energy_groups[group]->solution_old =
                energy_groups[group]->solution;
              energy_groups[group]->solution_old /= k_eff;
            }

          ++iteration;
        }

      while ((error > convergence_tolerance) && (iteration < 500));

      // Vector<double> group_l2_norm(n_groups);
      // Vector<double> group_linfty_norm(n_groups);

      {
        TimerOutput::Scope scope(timer_output, "Write output");
        for (unsigned int group = 0; group < n_groups; ++group)
        {
          energy_groups[group]->output_results(cycle);

          // if (error_mode == "fine")
          // {
          //   std::ofstream output("fine_solution-"
          //                         +  Utilities::int_to_string(group, 2)
          //                         + ".txt");
          //   for (unsigned int i = 0; i < energy_groups[group]->solution.size(); i++)
          //   {
          //     output << energy_groups[group]->solution[i] << std::endl;
          //   }
          // } else
          // {
          //   GridIn<dim> fine_grid;
          //   Triangulation<dim> fine_tria;
          //   fine_grid.attach_triangulation(fine_tria);
          //   fine_grid.read_msh(fine_grid_input);
          //   const DoFHandler<dim> fine_dof(fine_grid);
          //   std::ifstream fine_input("fine_solution-"
          //                            +  Utilities::int_to_string(group, 2)
          //                            + ".txt");
          //   Vector<double> fine_solution(fine_grid.n_active_cells())
          //   for (unsined int i = 0; i < fine_grid.n_active_cells(); i++)
          //   {
              
          //   }
          //   Functions::FEFieldFunction<dim> fine_function(fine_dof, fine_solution);
          //   group_l2_norm[group] = energy_groups[group]->compute_l2_norm();
          //   group_linfty_norm[group] = energy_groups[group]->compute_linfty_norm();
          // }
        }
      }

      PetscMemoryGetCurrentUsage(&memory);

      int total_dofs = 0;
      int total_cells = 0;
      for (unsigned int group = 0; group < n_groups; ++group)
        {  
          total_dofs += energy_groups[group]->n_dofs();
          total_cells += energy_groups[group]->n_active_cells();
        }

      std::cout << std::endl;
      std::cout << "Refinement Cycle      = " << cycle << std::endl
                << "Total DoFs            = " << total_dofs << std::endl
                << "Total Cells           = " << total_cells << std::endl
                << "Total Inner Iteration = " << inner_iteration << std::endl
                << "k_eff                 = " << k_eff << std::endl
                << "Time                  = " << timer.cpu_time() << " s" << std::endl
                << "Memory                = " << memory * 1e-6 << " MB" << std::endl;
      
      // for (unsigned int group = 0; group < n_groups; ++group)
      // {
      //   std::cout << "   Group " << group
      //             << ", L2 norm = " << group_l2_norm[group]
      //             << ", Linfty norm = " << group_linfty_norm[group]
      //             << std::endl;
      // }
      std::cout << std::endl;

      timer_output.print_summary();
      timer_output.reset();
    }

  }

template class StaticDiffusion<2>;
