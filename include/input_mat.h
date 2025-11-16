/**
 * @file   input/input_mat.h
 * @brief  InputMat class template declarations
 */
#ifndef FOREST_INPUT_MAT_H
#define FOREST_INPUT_MAT_H

// This is for the xml parser from boost

#include <map>
#include <string>
#include <vector>
#include <utility>

namespace XMLInput
{

/**
 * @brief Type of cross sections.
 *
 * @p XS_type is to know the cross sections needed for
 * this particular problem.
 *
 */
enum class XS_type
  : unsigned int
  {
    diffussion = 0, /**<  Diffusion cross sections. */
    transport = 1 /**<  Transport cross sections. */
};

/**
 * @brief Structure containing the data of a single material.
 *
 * Different data defining the properties of a single material,
 * and booleans variable to check which data has been provided
 * to this material or not.
 */
struct XS_single
{
  
  std::string name;
  std::vector<double> sigma_t;
  std::vector<double> nu_sigma_f;
  std::vector<double> chi;
  std::vector<std::vector<double> > sigma_s;
  std::vector<double> chi_p;

  
  bool exist_sigma_t = false;
  bool exist_sigma_tr = false;
  bool exist_nu_sigma_f = false;
  bool exist_sigma_r = false;
  bool exist_chi = false;
  bool exist_sigma_s = false;


  std::vector<double> sigma_a;
  std::vector<double> sigma_r;
  std::vector<double> sigma_tr;
  std::vector<double> nu;
  std::vector<double> sigma_f;

  
  bool exist_sigma_a = false;
  bool exist_nu = false;
  bool exist_sigma_f = false;

  unsigned int id;
};

/**
 * @class InputMat
 *
 * @brief Class for the cross sections.
 *
 * Here we have the cross sections for all the materials,
 * as well as some functions to read and write this data
 * to xml format, and some functions to check the data
 * when enough cross sections are available, i.e.,
 * we check that
 * \f$ \nu\Sigma_{f} = \nu*\Sigma_{f} \f$
 *  and we check
 * \f$ \Sigma_{t,g} = \sum_{h}(\Sigma_{s,g,h}) +
 * \Sigma_{a,g} \f$
 *
 * @todo The cross sections for the diffusion equation
 * should be added here.
 *
 */
class InputMat
{
public:

  
  typedef std::map<unsigned int, XS_single> XS_map;
  
  typedef std::pair<unsigned int, XS_single> XS_pair;

  
  XS_map xs;

  /**
   @brief Load the material data from the file @p filename
   @param filename
   */
  void
  load(const std::string &filename);

  /**
   @brief Print the material data to the file @p filename
   @param filename
   */
  void
  save(const std::string &filename);

  /**
   @brief Check Consistency of the materials data.
   @details check the data
   when enough cross sections are available, i.e.,
   we check that
   \f$ \nu\Sigma_{f} = \nu*\Sigma_{f} \f$
   and we check
   \f$ \Sigma_{t,g} = \sum_{h}(\Sigma_{s,g,h}) + \Sigma_{a,g} \f$
   */
  void
  check();

  /**
   * @brief Get the number of energy groups.
   * @return n_groups
   */
  unsigned int
  get_n_groups() const
  {
    return n_groups;
  }

  /**
   * @brief Get the number of different materials defined.
   * @return n_mat
   */
  unsigned int
  get_n_mat() const
  {
    return n_mat;
  }

  /**
   * @brief Get the number of different materials defined.
   * @return n_mat
   */
  std::vector<unsigned int>&
  get_materials_vector()
  {
    return materials_vector;
  }


private:

  unsigned int n_groups;
  unsigned int n_mat;
  unsigned int n_precursors;
  std::vector<unsigned int> materials_vector;

  
  void
  check_nusigf(XS_single & xs_) const;
  
  void
  calc_nusigf(XS_single & xs_);
  
  void
  check_sigmat(XS_single & xs_) const;
  
  void
  calc_sigmat(XS_single & xs_);
  
  void
  calc_chip(XS_single & xs_);
  
  void
  norm_chi(XS_single & xs_);
};

} // end of namespace XMLInput

#endif
