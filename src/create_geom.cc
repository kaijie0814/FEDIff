/**
 * @file   create_geom.cc
 * @brief
 */

// all include files you need here
#include <deal.II/base/types.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.templates.h>
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
#include <deal.II/fe/mapping_q.h>

#include <cmath>
#include <math.h>

#include "../include/input_geom.h"
#include "../include/create_geom.h"
#include "../include/utils.h"

using namespace dealii;

/**
 * Here the merge triangulation function to merge all the fuel pins
 * into a full assembly/ core.
 */
template <int dim>
void merge_triangulations(const unsigned int n_tria,
                          const std::vector<Point<2>> &centers,
                          const std::vector<Triangulation<2>> &tria_in,
                          Triangulation<2> &tria_result)
{

}


/**
 * Here the pin function. We assume the pin cell is a square of
 * length @p pitch with a circular subdomain of radius
 * @p pin_radius, centered at @p center.
 */
template <int dim>
void pin_rod(Triangulation<dim> &tria,
              const Point<2> &center,
              const double pitch,
              const double radius,
              const unsigned int inner_material,
              const unsigned int outer_material)
{

    Triangulation<dim> tria_inner;
    GridGenerator::hyper_ball_balanced(tria_inner, Point<dim>(), radius);
    // tria_inner.set_all_manifold_ids(1);
    for (const auto &cell : tria_inner.cell_iterators())
      cell->set_material_id(inner_material);
 
    Triangulation<dim> tria_outer;
    GridGenerator::hyper_cube_with_cylindrical_hole(
      tria_outer, radius, pitch/2.0, true);
    for (const auto &cell : tria_outer.cell_iterators())
      cell->set_material_id(outer_material);
 
    GridGenerator::merge_triangulations(tria_inner, tria_outer, tria);

    // Shift the pin to the desired center
    Tensor<1, dim> shift;
    GridTools::shift(center, tria);

    tria.reset_all_manifolds();
    tria.set_all_manifold_ids(0);

    const double r2 = radius*radius;
    const double tol = 1e-10;
    for (const auto &cell : tria.cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          {
            bool face_at_sphere_boundary = true;
            for (const auto v : face->vertex_indices())
              {
                if (std::abs((face->vertex(v)-center).norm_square() - r2) > tol)
                  {
                    face_at_sphere_boundary = false;
                    break;
                  }
              }
            if (face_at_sphere_boundary)
              face->set_all_manifold_ids(1);
          }
      }

    tria.set_manifold(1,SphericalManifold<dim>(center));
    // TransfiniteInterpolationManifold<dim> transfinite_manifold;
    // transfinite_manifold.initialize(tria);
    // tria.set_manifold(0, std::move(transfinite_manifold));
}

/**
 * Here the pin function. We assume the pin cell is a square of
 * length @p pitch with a circular subdomain of radius
 * @p pin_radius, centered at @p center.
 */
template <int dim>
void pin_box(Triangulation<dim> &tria,
            const Point<2> &center,
             const double pitch,
             const unsigned int first_mat_id)
{
    Point<dim> lower, upper;
    for (unsigned int d = 0; d < dim; ++d)
    {
        lower[d] = center[d] - pitch / 2.0;
        upper[d] = center[d] + pitch / 2.0;
    }
    GridGenerator::hyper_rectangle(tria, lower, upper);    

    for (const auto &cell : tria.cell_iterators())
      cell->set_material_id(first_mat_id);
}

/**
 * Calculate the effective radius to pass to the GridGenerator::pin_cell()
 * to maintain the fuel area.
 */
double calculate_eff_radius(const double radius)
{
  const double n_edges = std::pow(2, 2);
  const double alpha = 2 * M_PI / n_edges; // interior angle of the polygon
  const double circunscribed_radious = radius * std::sqrt(2 * M_PI / (n_edges * std::sin(alpha)));
  cout << circunscribed_radious << std::endl;

  return circunscribed_radious;
}


/**
 * Create the geometry
 */
template <int dim>
void make_grid(InputGeom &geom,
               Triangulation<dim> &tria_result,
               Materials &materials)
{
  assert(dim == 2 or dim == 3);
  unsigned int counter;
  const unsigned int nothing = static_cast<unsigned int>(-1);

  // pin_map
  const std::vector<std::vector<unsigned int>> pins_map = geom.planes[0].components;
  const std::vector<unsigned int> n_pins =
      {static_cast<unsigned int>(pins_map.size()),
       static_cast<unsigned int>(pins_map[0].size())};

  // How many not empty pins do we have
  unsigned int n_total_pins = 0;
  for (unsigned int i = 0; i < n_pins[0]; ++i)
    for (unsigned int j = 0; j < n_pins[1]; ++j)
      if (pins_map[i][j] != nothing)
        n_total_pins++;

  std::vector<Triangulation<2>> tria_pin(n_total_pins);
  Triangulation<2> tria_plane;

  // Now we work with dim = 2 to generate the geometry before extrusion
  std::vector<Point<2>> pin_center(n_total_pins);
  std::vector<double> pin_radius(n_total_pins);
  std::vector<Pin_type> pin_types(n_total_pins);

  // We generate isolated pins in a vector of triangulations, in order
  // to merge them later
  // It is assumed that all pins have the same pitch
  counter = 0;
  for (unsigned int j = 0; j < n_pins[1]; ++j)
  {
    for (unsigned int i = 0; i < n_pins[0]; ++i)
    {
      if (pins_map[i][j] != nothing)
      {
        InputGeom::Pin &pin = geom.pins[pins_map[i][j]];

        switch (pin.pin_type)
        {
          case Pin_type::pin:
            {
                pin_center[counter][0] = double(i) * pin.pitch + pin.pitch / 2;
                pin_center[counter][1] = double(j) * pin.pitch + pin.pitch / 2;
                //pin_radius[counter] = calculate_eff_radius(pin.fuel_radius);
                pin_types[counter] = Pin_type::pin;

                // Make pin triangulation
                pin_rod(tria_pin[counter],
                         pin_center[counter],
                         pin.pitch,
                         pin.fuel_radius,
                         //pin_radius[counter],
                         pin.materials[0],
                         pin.materials[1]);
                break;
            }
          case Pin_type::box:
            {
                pin_center[counter][0] = double(i) * pin.pitch + pin.pitch / 2;
                pin_center[counter][1] = double(j) * pin.pitch + pin.pitch / 2;
                pin_radius[counter] = 0.0;
                pin_types[counter] = Pin_type::box;

                // Make pin triangulation
                pin_box(tria_pin[counter],
                        pin_center[counter],
                        pin.pitch,
                        pin.materials[0]);
                break;
            }
            default:
            {
                Assert(false, ExcNotImplemented());
                break;
            }
        }
        counter++;
      }
    }
  }
  
  for (unsigned int i = 0; i < n_total_pins; i++)
  {
    GridGenerator::merge_triangulations(tria_pin[i], tria_result, tria_result);
  }

  // tria_result.reset_all_manifolds();

  
  // for (unsigned int i = 0; i < n_total_pins; i++)
  // {
  //   tria_result.set_all_manifold_ids(0);
  //   double radius = pin_radius[i];
  //   Point<2> center = pin_center[i];
  //   const double r2 = radius*radius;
  //   const double tol = 1e-10;
  //   for (const auto &cell : tria_result.cell_iterators())
  //     {
  //       for (const auto &face : cell->face_iterators())
  //         {
  //           bool face_at_sphere_boundary = true;
  //           for (const auto v : face->vertex_indices())
  //             {
  //               if (std::abs((face->vertex(v)-center).norm_square() - r2) > tol)
  //                 {
  //                   face_at_sphere_boundary = false;
  //                   break;
  //                 }
  //             }
  //           if (face_at_sphere_boundary)
  //             face->set_all_manifold_ids(1);
  //         }
  //     }

  //   tria_result.set_manifold(1,SphericalManifold<dim>(center));
    // TransfiniteInterpolationManifold<dim> transfinite_manifold;
    // transfinite_manifold.initialize(tria_result);
    // tria.set_manifold(0, transfinite_manifold);
    // tria_result.set_all_manifold_ids(0);

  // }
  //tria_result.set_manifold(1,SphericalManifold<dim>());

  // DataOut<dim> data_out;
  // DataOutBase::VtkFlags flags;
  // flags.write_higher_order_cells = true;
  // data_out.set_flags(flags);
  // data_out.attach_triangulation(tria_result);
  // const int output_mesh_order = 4;
  // MappingQ<dim> mapping = 4;
  // data_out.build_patches(mapping, output_mesh_order, DataOut<dim>::curved_inner_cells);
  // std::string vtk_filename = "pin_asmbly.vtk";
  // std::ofstream vtk_output(vtk_filename);
  // data_out.write_vtk(vtk_output); 
}

template void make_grid(InputGeom &geom,
                        Triangulation<2> &tria,
                        Materials &materials);