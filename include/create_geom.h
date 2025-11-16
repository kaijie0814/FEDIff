/**
 * @file   create_geom.h
 * @brief
 */

#ifndef CREATE_GEOM_H
#define CREATE_GEOM_H

// all include files you need here
#include <deal.II/base/types.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/base/parameter_handler.h>

#include <cmath>
#include <math.h>

#include "input_geom.h"
#include "materials.h"

using namespace dealii;


/**
 * Here the pin function. We assume the pin cell is a square of
 * length @p pitch with a circular subdomain of radius
 * @p pin_radius, centered at @p center.
 */
template <int dim>
  void pin_rod (Triangulation<dim> & /* tria */,
    const Point<2> & /* center */,
    const double /* pitch */,
    const double /* effective_pin_radius */,
    const unsigned int /*inner_material*/,
    const unsigned int /*outer_material*/);

template <int dim>
    void pin_box(Triangulation<dim> & /* tria */,
                const Point<2> & /* center */,
                const double /* pitch */,
                const unsigned int /* first_mat_id = 0 */);

/**
 * @brief Calculate the effective radius to pass to the GridGenerator::pin_cell()
 * to maintain the fuel area.
 */
double calculate_eff_radius (
  const double radius);


/**
 *
 */
template <int dim>
  void make_grid (InputGeom & geom,
    Triangulation<dim> &tria,
    Materials &materials);

#endif
