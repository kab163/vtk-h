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
#include <sstream>


std::map<std::string, std::string> args;

void checkValidity(vtkh::DataSet *data, const int maxSteps)
{
  int numDomains = data->GetNumberOfDomains();

  //Check all domains
  for(int i = 0; i < numDomains; i++)
  {
    auto currentDomain = data->GetDomain(i);
    vtkm::cont::CellSetExplicit<> cellSet =
          currentDomain.GetCellSet().Cast<vtkm::cont::CellSetExplicit<>>();

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

void LoadData(const std::string &fname, vtkh::DataSet &dataSet)
{
  int rank, nRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (fname == "")
  {
      const int base_size = 32;
      const int blocks_per_rank = 1;
      const int num_blocks = nRanks * blocks_per_rank;
      for(int i = 0; i < blocks_per_rank; ++i)
      {
          int domain_id = rank * blocks_per_rank + i;
          dataSet.AddDomain(CreateTestDataRectilinear(domain_id, num_blocks, base_size), domain_id);
      }
  }
  else
  {
      std::string buff;
      std::ifstream is;
      is.open(args["--filename"]);
      if (!is)
      {
          std::cout<<"File not found! : "<<args["--filename"]<<std::endl;
      }
      if (!is) throw "unknown file: " + args["--filename"];

      auto p0 = fname.rfind(".visit");
      if (p0 == std::string::npos)
          throw "Only .visit files are supported.";
      auto tmp = fname.substr(0, p0);
      auto p1 = tmp.rfind("/");
      auto dir = tmp.substr(0, p1);

      std::getline(is, buff);
      auto numBlocks = std::stoi(buff.substr(buff.find("!NBLOCKS ")+9, buff.size()));
      if (rank == 0) std::cout<<"numBlocks= "<<numBlocks<<std::endl;

      int nPer = numBlocks / nRanks;
      int b0 = rank*nPer, b1 = (rank+1)*nPer;
      if (rank == (nRanks-1))
          b1 = numBlocks;

      for (int i = 0; i < numBlocks; i++)
      {
          std::getline(is, buff);
          if (i >= b0 && i < b1)
          {
              vtkm::cont::DataSet ds;
              std::string vtkFile = dir + "/" + buff;
              vtkm::io::reader::VTKDataSetReader reader(vtkFile);
              ds = reader.ReadDataSet();
              int np = ds.GetNumberOfPoints();

              /*
              auto field = ds.GetField("grad").GetData();
              vtkm::cont::ArrayHandle<vtkm::Vec3f> vecField;
              field.CopyTo(vecField);
              auto portal = vecField.GetPortalControl();
              for (int j = 0; j < np; j++)
                  portal.Set(j, vtkm::Vec3f(1,0,0));
              */

              dataSet.AddDomain(ds, i);
          }
      }
  }
}

//----------------------------------------------------------------------------
TEST(vtkh_particle_advection, vtkh_serial_particle_advection)
{
  MPI_Init(NULL, NULL);
  int rank, nRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vtkh::SetMPICommHandle(MPI_Comm_c2f(MPI_COMM_WORLD));


  if (rank == 0)
  {
      std::cout << "Running parallel Particle Advection, vtkh - with " << nRanks << " ranks" << std::endl;
      //std::cout<<args<<std::endl;
  }

  vtkh::DataSet dataSet;

  LoadData(args["--filename"], dataSet);

  vtkh::ParticleAdvection streamline;
  streamline.SetGatherTraces((std::stoi(args["--streamline"]) == 1));
  streamline.SetDumpOutputFiles((std::stoi(args["--dump"]) == 1));
  streamline.SetInput(&dataSet);
  streamline.SetField(args["--field"]);

  streamline.SetMaxSteps(std::stoi(args["--maxSteps"]));
  streamline.SetStepSize(std::stof(args["--stepSize"]));
  streamline.SetSeedsRandomWhole(std::stoi(args["--numSeeds"]));
  streamline.SetUseThreadedVersion(std::stoi(args["--threaded"]));
  streamline.SetDelaySend(std::stoi(args["--delaySend"]));
  std::string device = args["--device"];

  if (rank == 0) std::cout<<"********************************** CUDAAvail= "<<vtkh::IsCUDAAvailable()<<std::endl;

  if (device == "serial")
      if (vtkh::IsSerialAvailable()) vtkh::ForceSerial();
      else throw "Serial device not available!";
  else if (device == "openmp")
      if (vtkh::IsOpenMPAvailable()) vtkh::ForceOpenMP();
      else throw "OpenMP device not available";
  else if (device == "cuda")
      if (vtkh::IsCUDAAvailable()) vtkh::ForceCUDA();
      else throw "CUDA device not available";

//  std::string dev = vtkh::GetCurrentDevice();
  if (rank == 0) std::cout<<" ***** Current Device= "<<vtkh::GetCurrentDevice()<<std::endl;

  if (args["--box"] != "")
  {
      std::string s, str = args["--box"];
      std::istringstream f(str);
      std::vector<double> vals;
      while (std::getline(f, s, ' '))
      {
          vals.push_back(std::stof(s));
      }
      vtkm::Bounds box(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]);
      streamline.SetSeedsRandomBox(std::stoi(args["--numSeeds"]), box);
  }
  if (args["--statsfile"] != "")
  {
      //mark it as started....
      std::string s = args["--statsfile"];
      FILE *fp = fopen(s.c_str(), "w");
      fprintf(fp, "Running...\n");
      fclose(fp);
      streamline.SetStatsFile(args["--statsfile"]);
  }
  if (args["--residentTime"] != "")
      streamline.SetResidentTimeDump(args["--residentTime"]);

  //streamline.SetSeedPoint(vtkm::Vec3f(1,1,1));
  streamline.SetBatchSize(std::stoi(args["--batchSize"]));

  streamline.Update();
  vtkh::DataSet *streamline_output = streamline.GetOutput();

  //checkValidity(streamline_output, maxAdvSteps);
  //writeDataSet(streamline_output, "advection_SeedsRandomWhole", rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}


int main(int argc, char* argv[])
{
    args["--filename"] = "";
    args["--field"] = "";
    args["--numSeeds"] = "100";
    args["--stepSize"] = "0.01";
    args["--maxSteps"] = "100";
    args["--batchSize"] = "-1";
    args["--streamline"] = "0";
    args["--dump"] = "0";
    args["--box"] = "";
    args["--threaded"] = "0";
    args["--statsfile"] = "";
    args["--device"] = "serial";
    args["--residentTime"] = "";
    args["--delaySend"] = "0";

    int i = 1;
    while (i < argc)
    {
        int inc = 1;
        std::string a0 = argv[i];
        std::string a1 = "1";
        if (i < argc-1)
        {
            std::string tmp(argv[i+1]);
            if (a0 == "--box")
            {
                a1 = "";
                for (int c = 0; c < 6; c++)
                    a1 = a1 + std::string(argv[i+1+c]) + " ";
                inc = 6;
                std::cout<<"box: "<<a1<<std::endl;
            }
            else if (tmp.find("--") == std::string::npos)
            {
                a1 = tmp;
                inc = 2;
            }

        }
        i+= inc;
        args[a0] = a1;
    }

/*
    for (auto m : args)
        std::cout<<"("<<m.first<<" "<<m.second<<")"<<std::endl;
*/

    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
//    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
//    MPI_Finalize();
    return result;
}
