#ifndef VTK_H_MARCHING_CUBES_HPP
#define VTK_H_MARCHING_CUBES_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class MarchingCubes : public Filter
{
public:
  MarchingCubes();
  virtual ~MarchingCubes();
  std::string GetName() const override;
  void SetIsoValue(const double &iso_value);
  void SetIsoValues(const double *iso_values, const int &num_values);
  void SetLevels(const int &levels);
  void SetField(const std::string &field_name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::vector<double> m_iso_values;
  std::string m_field_name;
  int m_levels;
};

} //namespace vtkh
#endif
