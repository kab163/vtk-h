#ifndef VTK_H_INTEGRATOR_HPP
#define VTK_H_INTEGRATOR_HPP

//#include "adapter.h"

#include <list>
#include <vector>
#include <deque>
#include <vector>
#include <string>

#include <vtkm/cont/DataSet.h>
#include <vtkm/worklet/ParticleAdvection.h>
#include <vtkm/worklet/particleadvection/GridEvaluators.h>
#include <vtkm/worklet/particleadvection/Integrators.h>
#include <vtkm/worklet/particleadvection/Particles.h>
#include <vtkm/worklet/particleadvection/ParticleAdvectionWorklets.h>
#include <vtkm/worklet/particleadvection/IntegratorStatus.h>

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Particle.hpp>
#include <vtkh/utils/ThreadSafeContainer.hpp>
#include <vtkh/StatisticsDB.hpp>

#ifdef VTKH_ENABLE_LOGGING
#define DBG(msg) vtkh::Logger::GetInstance("out")->GetStream()<<msg
#define WDBG(msg) vtkh::Logger::GetInstance("wout")->GetStream()<<msg
#else
#define DBG(msg)
#define WDBG(msg)
#endif


class ParticleAdvectWorkletORIG : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn idx,
                                ExecObject integrator,
                                ExecObject integralCurve,
                                FieldIn maxSteps);
  using ExecutionSignature = void(_1 idx, _2 integrator, _3 integralCurve, _4 maxSteps);
  using InputDomain = _1;

  template <typename IntegratorType, typename IntegralCurveType>
  VTKM_EXEC void operator()(const vtkm::Id& idx,
                            const IntegratorType* integrator,
                            IntegralCurveType& integralCurve,
                            const vtkm::Id& maxSteps) const
  {
    vtkm::Particle particle = integralCurve.GetParticle(idx);

    vtkm::Vec3f inpos = particle.Pos;
    vtkm::FloatDefault time = particle.Time;
    bool tookAnySteps = false;

    //the integrator status needs to be more robust:
    // 1. you could have success AND at temporal boundary.
    // 2. could you have success AND at spatial?
    // 3. all three?

    integralCurve.PreStepUpdate(idx);
    do
    {
      //vtkm::Particle p = integralCurve.GetParticle(idx);
      //std::cout<<idx<<": "<<inpos<<" #"<<p.NumSteps<<" "<<p.Status<<std::endl;
      vtkm::Vec3f outpos;
      auto status = integrator->Step(inpos, time, outpos);
      if (status.CheckOk())
      {
        integralCurve.StepUpdate(idx, time, outpos);
        tookAnySteps = true;
        inpos = outpos;
      }

      //We can't take a step inside spatial boundary.
      //Try and take a step just past the boundary.
      else if (status.CheckSpatialBounds())
      {
        auto status2 = integrator->SmallStep(inpos, time, outpos);
        if (status2.CheckOk())
        {
          integralCurve.StepUpdate(idx, time, outpos);
          tookAnySteps = true;

          //we took a step, so use this status to consider below.
          status = status2;
        }
      }

      integralCurve.StatusUpdate(idx, status, maxSteps);

    } while (integralCurve.CanContinue(idx));

    //Mark if any steps taken
    integralCurve.UpdateTookSteps(idx, tookAnySteps);

    //particle = integralCurve.GetParticle(idx);
    //std::cout<<idx<<": "<<inpos<<" #"<<particle.NumSteps<<" "<<particle.Status<<std::endl;
  }
};


class VTKH_API Integrator
{
    typedef vtkm::Float64 FieldType;
    typedef vtkm::cont::ArrayHandle<vtkm::Vec<FieldType, 3>> FieldHandle;

    using GridEvalType = vtkm::worklet::particleadvection::GridEvaluator<FieldHandle>;
    using RK4Type = vtkm::worklet::particleadvection::RK4Integrator<GridEvalType>;

public:
    Integrator(vtkm::cont::DataSet *ds, const std::string &fieldName, FieldType _stepSize, int _batchSize, int _rank)
      : stepSize(_stepSize), batchSize(_batchSize), rank(_rank)
    {
        vecField = ds->GetField(fieldName).GetData().Cast<FieldHandle>();
        gridEval = GridEvalType(ds->GetCoordinateSystem(), ds->GetCellSet(), vecField);
        rk4 = RK4Type(gridEval, stepSize);
    }

    int Advect(std::vector<vtkh::Particle> &particles,
               vtkm::Id MaxSteps,
               std::vector<vtkh::Particle> &I,
               std::vector<vtkh::Particle> &T,
               std::vector<vtkh::Particle> &A,
               std::vector<vtkm::worklet::ParticleAdvectionResult> *particleTraces,
               vtkh::ThreadSafeContainer<vtkh::Particle, std::vector> &workerInactive,
               vtkh::StatisticsDB& statsDB,
               bool delaySend=false);

    int Advect(std::vector<vtkh::Particle> &particles,
               vtkm::Id maxSteps,
               std::vector<vtkh::Particle> &I,
               std::vector<vtkh::Particle> &T,
               std::vector<vtkh::Particle> &A,
               vtkh::StatisticsDB& statsDB,
               std::vector<vtkm::worklet::ParticleAdvectionResult> *particleTraces=NULL)
    {
        size_t nSeeds = particles.size();
        vtkm::cont::ArrayHandle<vtkm::Particle> seedArray;

        int steps0 = SeedPrep(particles, seedArray);

#ifdef VTKH_USE_CUDA
    // This worklet needs some extra space on CUDA.
    vtkm::cont::cuda::ScopedCudaStackSize stack(16 * 1024);
    (void)stack;
#endif
        vtkm::worklet::ParticleAdvection particleAdvection;
        vtkm::worklet::ParticleAdvectionResult result;

        vtkh::StopWatch residentTimer;
        residentTimer.Start();
#if 1
        using ParticleType = vtkm::worklet::particleadvection::Particles;
        vtkm::Id numSeeds = static_cast<vtkm::Id>(seedArray.GetNumberOfValues());
        ParticleAdvectWorkletORIG worklet;
        ParticleType particlesObj(seedArray, maxSteps);
        vtkm::cont::ArrayHandleConstant<vtkm::Id> maxStepArr(maxSteps, numSeeds);
        vtkm::cont::ArrayHandleIndex idxArray(numSeeds);

        vtkm::worklet::DispatcherMapField<ParticleAdvectWorkletORIG> dispatcher(worklet);
        TIMER_START("residentTime");
        dispatcher.Invoke(idxArray, rk4, particlesObj, maxStepArr);
        TIMER_STOP("residentTime");
        result = vtkm::worklet::ParticleAdvectionResult(seedArray);
#else
        TIMER_START("residentTime");
        result = particleAdvection.Run(rk4, seedArray, maxSteps);
        TIMER_STOP("residentTime");
#endif

        double residentTime = residentTimer.Stop();
        auto parPortal = result.Particles.GetPortalConstControl();

        //Update particle data.
        //Need a functor to do this...
        int steps1 = 0;
        for (int i = 0; i < nSeeds; i++)
        {
            particles[i].p = parPortal.Get(i);
            particles[i].p.ResidentTime += residentTime;
            UpdateParticle(particles[i], I,T,A);
            steps1 += particles[i].p.NumSteps;
        }

        /*
        {
            for (int i = 0; i < nSeeds; i++)
            {
                vtkm::Vec<FieldType,4> p(posPortal.Get(i)[0],
                                     posPortal.Get(i)[1],
                                     posPortal.Get(i)[2],
                                     particles[i].id);
//                                     particles[i].blockId);
                particleTraces->push_back(p);
            }
        }
        */

        if (particleTraces)
          (*particleTraces).push_back(result);

        int totalSteps = steps1-steps0;
        return totalSteps;
    }

    int Trace(std::vector<vtkh::Particle> &particles,
              vtkm::Id maxSteps,
              std::vector<vtkh::Particle> &I,
              std::vector<vtkh::Particle> &T,
              std::vector<vtkh::Particle> &A,
              std::vector<vtkm::worklet::StreamlineResult> *particleTraces=NULL)
    {
        size_t nSeeds = particles.size();
        vtkm::cont::ArrayHandle<vtkm::Particle> seedArray;

        int steps0 = SeedPrep(particles, seedArray);

        vtkm::worklet::Streamline streamline;
        vtkm::worklet::StreamlineResult result;
        result = streamline.Run(rk4, seedArray, maxSteps);
        auto parPortal = result.Particles.GetPortalConstControl();

        //Update particle data.
        int steps1 = 0;
        for (int i = 0; i < nSeeds; i++)
        {
            /*
            vtkm::cont::ArrayHandle<vtkm::Id> ids;
            result.PolyLines.GetIndices(i, ids);
            auto idPortal = ids.GetPortalConstControl();
            vtkm::Id nPts = idPortal.GetNumberOfValues();
            */

            particles[i].p = parPortal.Get(i);
            steps1 += particles[i].p.NumSteps;

            UpdateParticle(particles[i], I,T,A);

#if 0
            {
                for (int j = 0; j < nPts; j++)
                {
                    vtkm::Vec3f p(parPortal.Get(idPortal.Get(j))[0],
                                  parPortal.Get(idPortal.Get(j))[1],
                                  parPortal.Get(idPortal.Get(j))[2]);
//                                         particles[i].id);
//                                         particles[i].blockId);

                    particleTraces->push_back(p);
                }
            }
#endif

        }
        if (particleTraces)
          (*particleTraces).push_back(result);

        int totalSteps = steps1-steps0;
        return totalSteps;
    }

private:

    void UpdateParticle(vtkh::Particle &p,
                        std::vector<vtkh::Particle> &I,
                        std::vector<vtkh::Particle> &T,
                        std::vector<vtkh::Particle> &A)
    {
      if (p.p.Status.CheckTerminate())
        T.push_back(p);
      else if (p.p.Status.CheckSpatialBounds())
        I.push_back(p);
      else if (p.p.Status.CheckOk())
        A.push_back(p);
      else
        T.push_back(p);
    }

    int SeedPrep(const std::vector<vtkh::Particle> &particles,
                 vtkm::cont::ArrayHandle<vtkm::Particle> &seedArray)
    {
        int stepsTaken = 0;
        size_t nSeeds = particles.size();
        seedArray.Allocate(nSeeds);
        auto seedPortal = seedArray.GetPortalControl();
        for (int i = 0; i < nSeeds; i++)
        {
            seedPortal.Set(i, particles[i].p);
            stepsTaken += particles[i].p.NumSteps;
        }
        return stepsTaken;
    }

    int rank;
    int batchSize;
    FieldType stepSize;
    GridEvalType gridEval;
    RK4Type rk4;
    FieldHandle vecField;
};

#endif
