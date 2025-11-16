/**
 * @file   materials.cc
 * @brief  Implementation of class Materials
 */

#include <boost/version.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "../include/input_mat.h"
#include "../include/input_complex_pert.h"
#include "../include/materials.h"
#include "../include/utils.h"
#include "../include/utils_base.h"

using namespace dealii;

/**
 *
 */
Materials::Materials (ConditionalOStream &verbose_cout) :
    verbose_cout(verbose_cout)
{
  n_groups = 0;
  n_mats = 0;
  listen_to_material_id = false;
  n_mats_init = 0;
  n_total_assemblies = 0;
  keff = 1.0;
  n_assemblies = 0;
}

/**
 *
 */
void Materials::reinit (const std::string &xsec_file,
  const unsigned int n_groups,
  const std::vector<unsigned int> &n_assemblies_per_dim,
  unsigned int &n_assem,
  bool listen_to_material_id)
{
  this->n_groups = n_groups;
  this->listen_to_material_id = listen_to_material_id;

  assem_per_dim = n_assemblies_per_dim;
  n_assemblies = n_assem;
  AssertRelease(fexists(xsec_file), "XSECS_Filename does not exist");

  // XML type file
  parse_forest_xs(xsec_file);

  const unsigned int per_plane = n_assem / n_assemblies_per_dim[2];
  assemblies_per_plane.resize(n_assemblies_per_dim[2], per_plane);

  n_total_assemblies = n_assemblies;

  // Delete dummy materials
  std::vector<unsigned int> temp;
  temp.reserve(materials_vector.size());
  for (unsigned int i = 0; i < materials_vector.size(); i++)
  {
    if (materials_vector[i] != static_cast<unsigned int>(-1))
      temp.push_back(materials_vector[i]);
    else
    {
      verbose_cout << "Hole in position: " << i << " in plane "
                   << i / per_plane
                   << std::endl;
      assemblies_per_plane[i / per_plane]--;
      n_assemblies--;
    }
  }

  // Remove materials with holes
  materials_vector_with_holes = materials_vector;
  materials_vector_no_bar = materials_vector;
  materials_vector = temp;
}




/**
 * @brief Get the number of energy groups.
 * @return n_groups
 */
unsigned int Materials::get_n_groups () const
{
  return n_groups;
}

/**
 * @brief Get the number of materials defined.
 * @return n_mats
 */
unsigned int Materials::get_n_mats () const
{
  return n_mats;
}

/**
 * @brief Get a constant reference to geometry_matrix
 */
unsigned int Materials::get_n_assemblies () const
{
  return materials_vector.size();
}


/**
 *
 */
const std::vector<unsigned int>& Materials::get_materials_vector () const
{
  return materials_vector;
}

/**
 *
 */
template <int dim>
  unsigned int Materials::get_material_id (
    typename DoFHandler<dim>::cell_iterator &cell) const
  {
    if (listen_to_material_id)
    {
      return cell->material_id();
    }
    else
    {
      AssertIndexRange(cell->user_index(), materials_vector.size());
      return materials_vector[cell->user_index()];
    }
  }

template unsigned int Materials::get_material_id<2> (
  typename DoFHandler<2>::cell_iterator&) const;

/**
 *
 */
template <int dim>
  unsigned int Materials::get_original_material_id (
    typename DoFHandler<dim>::active_cell_iterator &cell) const
  {
    AssertIndexRange(cell->user_index(), materials_vector_no_bar.size());
    return materials_vector_no_bar[cell->user_index()];
  }

template
unsigned int Materials::get_original_material_id<2> (
  typename DoFHandler<2>::active_cell_iterator&) const;

/**
 *
 */
void Materials::set_materials_id (
  const unsigned int cell_user_index,
  const unsigned int mat_id)
{
  AssertIndexRange(cell_user_index, materials_vector.size());
  AssertIndexRange(mat_id, n_mats);
  materials_vector[cell_user_index] = mat_id;
}

/**
 *
 */
double Materials::get_sigma_tr (
  unsigned int group,
  unsigned int mat) const
{
  AssertIndexRange(group, sigma_t.size());
  AssertIndexRange(mat, sigma_t[group].size());
  return sigma_tr[group][mat];
}

/**
 *
 */
double Materials::get_sigma_t (
  unsigned int group,
  unsigned int mat) const
{
  AssertIndexRange(group, sigma_t.size());
  AssertIndexRange(mat, sigma_t[group].size());
  return sigma_t[group][mat];

}

/**
 *
 */
std::vector<double> Materials::get_sigma_tr (const unsigned int group) const
{
  AssertIndexRange(group, sigma_tr.size());
  return sigma_tr[group];
}

/**
 *
 */
double Materials::get_chi (
  unsigned int group,
  unsigned int mat) const
{
  AssertIndexRange(group, chi.size());
  AssertIndexRange(mat, chi[group].size());
  return chi[group][materials_vector[mat]];
}

/**
 *
 */
double Materials::get_sigma_r (
  const unsigned int group,
  const unsigned int mat) const
{
  AssertIndexRange(group, sigma_r.size());
  AssertIndexRange(mat, sigma_r[group].size());
  return sigma_r[group][mat];
}

/**
 *
 */
std::vector<double> Materials::get_sigma_r (const unsigned int group) const
{
  AssertIndexRange(group, sigma_r.size());
  return sigma_r[group];
}

/**
 *
 */
std::vector<double> Materials::get_sigma_f (const unsigned int group) const
{
  AssertIndexRange(group, sigma_f.size());
  return sigma_f[group];
}

/**
 *
 */
std::vector<double> Materials::get_sigma_s (
  const unsigned int group_i,
  const unsigned int group_j) const
{
  AssertIndexRange(group_i, sigma_s.size());
  return sigma_s[group_i][group_j];
}

/**
 *
 */
double Materials::get_xi_nu_sigma_f (
  const unsigned int from_group,
  const unsigned int to_group,
  const unsigned int mat) const
{
  AssertIndexRange(to_group, chi.size());
  AssertIndexRange(from_group, nu_sigma_f.size());
  AssertIndexRange(mat, chi[to_group].size());
  AssertIndexRange(mat, nu_sigma_f[from_group].size());

  return chi[to_group][mat] * nu_sigma_f[from_group][mat];
}

/**
 *
 */
double Materials::get_nu_sigma_f (
  const unsigned int group,
  const unsigned int mat) const
{

  AssertIndexRange(group, nu_sigma_f.size());
  AssertIndexRange(mat, nu_sigma_f[group].size());

  return nu_sigma_f[group][mat];
}

/**
 *
 */
double Materials::get_sigma_f (const unsigned int group,
  const unsigned int mat) const
{
  AssertIndexRange(group, sigma_f.size());
  AssertIndexRange(mat, sigma_f[group].size());
  return sigma_f[group][mat];
}

/**
 *
 */
std::vector<double> Materials::get_xi_nu_sigma_f (const unsigned int from_group,
  const unsigned int to_group) const
{
  AssertIndexRange(to_group, chi.size());
  AssertIndexRange(from_group, nu_sigma_f.size());

  std::vector<double> chinusigmaf(n_mats);
  for (unsigned int nm = 0; nm < n_mats; nm++)
    chinusigmaf[nm] = chi[to_group][nm] * nu_sigma_f[from_group][nm];

  return chinusigmaf;
}

/**
 *
 */
std::vector<double> Materials::get_nu_sigma_f (const unsigned int group) const
{

  std::vector<double> nusigmaf(n_mats);
  for (unsigned int nm = 0; nm < n_mats; nm++)
    nusigmaf[nm] = nu_sigma_f[group][nm];

  return nusigmaf;
}

/**
 *
 */
double Materials::get_sigma_s (
  const unsigned int from_group,
  const unsigned int to_group,
  const unsigned int mat) const
{
  AssertIndexRange(from_group, sigma_s.size());
  AssertIndexRange(to_group, sigma_s[from_group].size());
  AssertIndexRange(mat, sigma_s[from_group][to_group].size());
  return sigma_s[from_group][to_group][mat];
}

/**
 *
 */
double Materials::get_diffusion_coefficient (
  const unsigned int group,
  const unsigned int mat) const
{
  AssertIndexRange(group, sigma_tr.size());
  AssertIndexRange(mat, sigma_tr[group].size());
  return 1 / (3 * sigma_tr[group][mat]);
}

void Materials::set_sigma_f (
  const double sigma_f_coeff,
  const unsigned int group,
  const unsigned int mat)
{
  AssertIndexRange(group, sigma_f.size());
  AssertIndexRange(mat, sigma_f[group].size());
  sigma_f[group][mat] = sigma_f_coeff;
}

/**
 *
 */
void Materials::set_nu_sigma_f (
  const double nu_sigma_f_coeff,
  const unsigned int group,
  const unsigned int mat)
{
  AssertIndexRange(group, nu_sigma_f.size());
  AssertIndexRange(mat, nu_sigma_f[group].size());
  nu_sigma_f[group][mat] = nu_sigma_f_coeff;
}

/**
 *
 */
void Materials::set_chi (
  const double chi_coeff,
  const unsigned int group,
  const unsigned int mat)
{
  AssertIndexRange(group, chi.size());
  AssertIndexRange(mat, chi[group].size());
  chi[group][mat] = chi_coeff;
}

/**
 *
 */
void Materials::set_sigma_s (const double set_sigma_s_coeff,
  const unsigned int from_group,
  const unsigned int to_group,
  const unsigned int mat)
{
  AssertIndexRange(from_group, sigma_s.size());
  AssertIndexRange(to_group, sigma_s[from_group].size());
  AssertIndexRange(mat, sigma_s[from_group][to_group].size());
  sigma_s[from_group][to_group][mat] = set_sigma_s_coeff;
}

/**
 *
 */
void Materials::set_sigma_r (const double sigma_r_coeff,
  unsigned int group,
  unsigned int mat)
{
  AssertIndexRange(group, sigma_r.size());
  AssertIndexRange(mat, sigma_r[group].size());
  sigma_r[group][mat] = sigma_r_coeff;
}

/**
 *
 */
void Materials::set_sigma_tr (
  const double sigma_tr_coeff,
  unsigned int group,
  unsigned int mat)
{
  AssertIndexRange(group, sigma_tr.size());
  AssertIndexRange(mat, sigma_tr[group].size());
  sigma_tr[group][mat] = sigma_tr_coeff;
}

/**
 *
 */
void Materials::set_sigma_t (
  const double sigma_t_coeff,
  unsigned int group,
  unsigned int mat)
{
  AssertIndexRange(group, sigma_t.size());
  AssertIndexRange(mat, sigma_t[group].size());
  sigma_t[group][mat] = sigma_t_coeff;
}


/**
 * @brief Make the reactor critical.
 */
void Materials::make_critical (const double &keffective)
{

  // Compute diffusion Coefficients
  for (unsigned int g = 0; g < n_groups; ++g)
    for (unsigned int mat = 0; mat < n_mats; ++mat)
    {
      nu_sigma_f[g][mat] /= keffective;
      sigma_f[g][mat] /= keffective;
    }

  keff = keffective;

}



/**
 *  @brief It parses the XS.xml file with a XML format.
 */
void Materials::parse_forest_xs (const std::string &xml_file)
{
  verbose_cout << "parse_forest_xs...  " << xml_file << std::endl;
  AssertRelease(fexists(xml_file), "forest_file doesn't exist");

  XMLInput::InputMat input;
  input.load(xml_file);
  AssertRelease(input.get_n_groups() == n_groups,
    "n_groups in xml file does not match " + num_to_str(n_groups)
    + " vs "
    + num_to_str(input.get_n_groups()));

  // Resize containers
  n_mats = input.get_n_mat();
  verbose_cout << "n_mats: " << n_mats << std::endl;

  // Resize Containers
  sigma_tr.resize(n_groups, std::vector<double>(n_mats));
  sigma_t.resize(n_groups, std::vector<double>(n_mats));
  sigma_s.resize(n_groups,
    std::vector<std::vector<double> >(n_groups,
      std::vector<double>(n_mats)));
  chi.resize(n_groups, std::vector<double>(n_mats));
  sigma_r.resize(n_groups, std::vector<double>(n_mats));
  nu_sigma_f.resize(n_groups, std::vector<double>(n_mats));
  sigma_f.resize(n_groups, std::vector<double>(n_mats));


  // Fill materials Vector
  materials_vector = input.get_materials_vector();
  if (materials_vector.empty())
    for (unsigned int mat = 0; mat < n_mats; ++mat)
      materials_vector.push_back(mat);

  for (unsigned int mat = 0; mat < n_mats; ++mat)
  {
    verbose_cout << "  Material " << mat << std::endl;
    AssertRelease(input.xs[mat].id == mat, "Error in mat ids");
    for (unsigned int from_g = 0; from_g < n_groups; ++from_g)
    {
      verbose_cout << "    Group " << from_g + 1 << std::endl;
      AssertRelease(input.xs[mat].exist_sigma_t,
        "Sigma_t does not exist");
      AssertRelease(input.xs[mat].exist_sigma_s,
        "Sigma_s does not exist");
      AssertRelease(input.xs[mat].exist_chi, "Chi does not exist");
      AssertRelease(input.xs[mat].exist_nu_sigma_f,
        "Nu Sigma_f does not exist");

      // sigma_t
      sigma_t[from_g][mat] = input.xs[mat].sigma_t[from_g]; // TODO
      verbose_cout << "        sigma_t_" << from_g + 1 << " = "
                   << sigma_tr[from_g][mat]
                   << std::endl;

      // sigma_tr
      if (input.xs[mat].exist_sigma_tr)
        sigma_tr[from_g][mat] = input.xs[mat].sigma_tr[from_g];
      else
        sigma_tr[from_g][mat] = input.xs[mat].sigma_t[from_g];

      verbose_cout << "        sigma_tr_" << from_g + 1 << " = "
                   << sigma_tr[from_g][mat]
                   << std::endl;

      // chi
      chi[from_g][mat] = input.xs[mat].chi[from_g];
      verbose_cout << "        chi_" << from_g + 1 << " = "
                   << chi[from_g][mat]
                   << std::endl;

      // nu_sigma_f
      nu_sigma_f[from_g][mat] = input.xs[mat].nu_sigma_f[from_g];
      verbose_cout << "        nu_sigma_fv_" << from_g + 1 << " = "
                   << nu_sigma_f[from_g][mat]
                   << std::endl;

      // sigma_r
      sigma_r[from_g][mat] = input.xs[mat].sigma_r[from_g];
      verbose_cout << "        nu_sigma_fv_" << from_g + 1 << " = "
                   << nu_sigma_f[from_g][mat]
                   << std::endl;

      // sigma_f
      // If sigma_f exists get it if not use nusigf instead
      if (!input.xs[mat].exist_sigma_f)
      {
        input.xs[mat].sigma_f = input.xs[mat].nu_sigma_f;
        input.xs[mat].exist_sigma_f = true;
      }
      sigma_f[from_g][mat] = input.xs[mat].sigma_f[from_g];
      verbose_cout << "        sigma_f_" << from_g + 1 << " = "
                   << sigma_f[from_g][mat]
                   << std::endl;

      // sigma_s
      for (unsigned int to_g = 0; to_g < n_groups; ++to_g)
      {
        // Be careful because in input.xs sigma_s is in a different way
        // Also be careful because we define sigma_s negative!
        sigma_s[from_g][to_g][mat] =
                                     input.xs[mat].sigma_s[to_g][from_g];

        verbose_cout << "        sigma_s_intergroup_" << from_g + 1
                     << "->"
                     << to_g + 1 << " = "
                     << sigma_s[from_g][to_g][mat]
                     << std::endl;
      }
    }

    double sum = 0;
    for (unsigned int to_g = 0; to_g < n_groups; ++to_g)
    {
      sum += chi[to_g][mat];
    }
  }

  verbose_cout << "materials_vector: " << std::flush;
  if (verbose_cout.is_active())
    print_vector(materials_vector, false);
  verbose_cout << " Done!" << std::endl;
}


/**
 * @brief Get the materials table as materials_table valid to
 */
template <int dim>
  void Materials::get_materials_table (
    Table<dim, types::material_id> &materials_table,
    const std::vector<unsigned int> &n_assemblies_per_dim)
  {
    materials_table = build_materials_table<dim>(
      geometry_matrix,
      n_assemblies_per_dim,
      materials_vector_with_holes);
  }


template void Materials::get_materials_table<2> (
  Table<2, types::material_id> &materials_table,
  const std::vector<unsigned int> &n_assemblies_per_dim);


/*
 * This should NOT be used, only defined to specialization below:
 */
template <int dim>
  Table<dim, types::material_id> Materials::build_materials_table (
    const std::vector<std::vector<unsigned int> > &geometry_matrix,
    const std::vector<unsigned int> &n_cells_per_dim,
    const std::vector<unsigned int> &materials)

  {
    AssertRelease(false, "setMaterialsTable ExcImpossibleInDim 1");
    Table<dim, types::material_id> materials_table;
    return materials_table;
  }


/*
 * Template Specialization for dim=2
 *  Fill materials_table a matrix that indicate which cell exist (1) or not (-1)
 */
template <>
  Table<2, types::material_id> Materials::build_materials_table<2> (
    const std::vector<std::vector<unsigned int> > &geometry_matrix,
    const std::vector<unsigned int> &n_cells_per_dim,
    const std::vector<unsigned int> &materials)
  {
    Table<2, types::material_id> materials_table(n_cells_per_dim[0],
      n_cells_per_dim[1]);

    unsigned int mat = 0;
    for (unsigned int j = 0; j < n_cells_per_dim[1]; j++)
      for (unsigned int i = 0; i < n_cells_per_dim[0]; i++)
      {
        if (geometry_matrix[j][i] != 0)
        {
          materials_table[i][j] = static_cast<types::material_id>(materials[mat]);
          mat++;
        }
        else
          materials_table[i][j] = static_cast<types::material_id>(-1); // Hole
      }
    return materials_table;
  }


/**
 * @brief Check if the cell in position (pos_x, pos_y, pos_z) is fuel.
 * It also advances an index if the cell is a hole or the reflector.
 */
bool Materials::is_fuel (const unsigned int pos_x,
  const unsigned int pos_y,
  const unsigned int,
  unsigned int &index) const
{
  if (listen_to_material_id)
    return true;
  if (geometry_matrix[pos_y][pos_x] == 0)
  {
    return false;
  }

  else if (materials_vector_with_holes[index]
           == static_cast<unsigned int>(-1))
  {
    index++;
    return false;
  }

  return true;
}

/**
 * @brief Check if the cell in position (pos_x, pos_y, pos_z) is reflector.
 */
bool Materials::is_reflector (const unsigned int pos_x,
  const unsigned int pos_y,
  const unsigned int) const
{
  return (geometry_matrix[pos_y][pos_x] == 2);
}

/**
 * @brief Return the plane number where the cell_index pertains
 */
unsigned int Materials::plane (int cell_index) const
{
  unsigned int plane = 0;
  while (cell_index >= 0)
  {
    cell_index -= assemblies_per_plane[plane];
    plane++;
  }
  return plane - 1;
}

/**
 * @brief Set geometry_matrix from the geometry_points structure.
 */
void Materials::set_geometry_matrix (
  const std::vector<unsigned int> &assem_per_dim,
  const std::vector<unsigned int> &geo_ps)
{
  geometry_matrix.resize(assem_per_dim[1],
    std::vector<unsigned int>(assem_per_dim[0], 0));
  for (unsigned int i = 0; i < assem_per_dim[1]; ++i)
    for (unsigned int j = geo_ps[2 * i] - 1; j < geo_ps[2 * i + 1]; ++j)
      geometry_matrix[i][j] = 1;
}

/**
 * @brief Set geometry_matrix from a string (from the input file)
 */
void Materials::set_geometry_matrix (
  const std::vector<unsigned int> &assem_per_dim,
  const std::string &str)
{
  parse_matrix(str, geometry_matrix, assem_per_dim[1], assem_per_dim[0]);
}

/**
 * @brief Get a constant reference to geometry_matrix.
 * @ return geometry matrix
 */
const std::vector<std::vector<unsigned int> >& Materials::get_geometry_matrix () const
{
  return geometry_matrix;
}

/**
 * @brief Check if the material in cell position (idx_x, idx_y, idx_z) exists.
 * @return true or false
 */
bool Materials::exist (
  const unsigned int idx_x,
  const unsigned int idx_y,
  const unsigned int) const
{
  return (geometry_matrix[idx_y][idx_x] != 0);
}
