//-----------------------------------------------------------------------------
///
/// file: t_vtk-h_particle_advection_par.cpp
///
//-----------------------------------------------------------------------------
#include "gtest/gtest.h"

#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/io/reader/VTKDataSetReader.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include "t_test_utils.hpp"
#include <iostream>
#include <mpi.h>

void checkValidity(vtkh::DataSet *data, const int maxSteps)
{
  int numDomains = data->GetNumberOfDomains();

  //Check all domains
  for(int i = 0; i < numDomains; i++)
  {
    auto currentDomain = data->GetDomain(i);
    vtkm::cont::CellSetExplicit<> cellSet =
          currentDomain.GetCellSet(0).Cast<vtkm::cont::CellSetExplicit<>>();

    //Ensure that streamlines took <= to the max number of steps
    for(int j = 0; j < cellSet.GetNumberOfCells(); j++)
    {
      EXPECT_LE(cellSet.GetNumberOfPointsInCell(j), maxSteps);
    }
  }
}

void writeDataSet(vtkh::DataSet *data, std::string fName, int rank)
{
  int numDomains = data->GetNumberOfDomains();
  std::cerr << "num domains " << numDomains << std::endl;
  for(int i = 0; i < numDomains; i++)
  {
    char fileNm[128];
    sprintf(fileNm, "%s.rank%d.domain%d.vtk", fName.c_str(), rank, i);
    vtkm::io::writer::VTKDataSetWriter write(fileNm);
    write.WriteDataSet(data->GetDomain(i));
  }
}

//----------------------------------------------------------------------------
TEST(vtkh_particle_advection, vtkh_serial_particle_advection)
{
  MPI_Init(NULL, NULL);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));

  std::cout << "Running parallel Particle Advection, vtkh - with " << comm_size << " ranks" << std::endl;

  vtkh::DataSet data_set;
  const int base_size = 32;
  const int blocks_per_rank = 1;
  const int maxAdvSteps = 1000;
  const int num_blocks = comm_size * blocks_per_rank;
  
  std::string fieldName = "vector_data_Float64";
  if(0) {
    for(int i = 0; i < blocks_per_rank; ++i)
    {
      int domain_id = rank * blocks_per_rank + i;
      data_set.AddDomain(CreateTestDataRectilinear(domain_id, num_blocks, base_size), domain_id);
    }
  } else {
    fieldName = "grad";
    char fname[64];
    int dom = rank;
    if (comm_size == 2)
        dom = (rank == 0 ? 0 : 3);

    //This is so that I can run on my laptop - ensures that 4 ranks together get the bottom half of domain
    if (rank == 0)
      sprintf(fname, "/Users/1a8/Documents/vtkh_build/fish8/fish_8.%01d.vtk", 1);
    else if (rank == 1)
      sprintf(fname, "/Users/1a8/Documents/vtkh_build/fish8/fish_8.%01d.vtk", 2);
    else if (rank == 2)
      sprintf(fname, "/Users/1a8/Documents/vtkh_build/fish8/fish_8.%01d.vtk", 4);
    else if (rank == 3)
      sprintf(fname, "/Users/1a8/Documents/vtkh_build/fish8/fish_8.%01d.vtk", 6);

    //sprintf(fname, "/Users/1a8/Documents/vtkh_build/fish8/fish_8.%01d.vtk", dom);

    std::cout<<"LOADING: "<<fname<<std::endl;
    vtkm::io::reader::VTKDataSetReader reader(fname);
    auto ds = reader.ReadDataSet();
    vtkm::cont::ArrayHandle<vtkm::Vec<double,3>> field;
    ds.GetField(fieldName).GetData().CopyTo(field);
    auto fportal = ds.GetField(fieldName).GetData();
    int nVecs = fportal.GetNumberOfValues();
    data_set.AddDomain(ds, rank);
  }

  vtkh::ParticleAdvection streamline;
  streamline.SetInput(&data_set);
  streamline.SetField(fieldName);
  streamline.SetMaxSteps(maxAdvSteps);
  streamline.SetStepSize(0.0001);
  streamline.SetSeedsRandomWhole(1000);
  streamline.SetUseThreadedVersion(true);
  streamline.SetDumpOutputFiles(false);
  streamline.SetGatherTraces(true);
  streamline.Update();
  vtkh::DataSet *streamline_output = streamline.GetOutput();

  checkValidity(streamline_output, maxAdvSteps);
  writeDataSet(streamline_output, "advection_SeedsRandomWhole", rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
