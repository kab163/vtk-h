#include <vtkh/StatisticsDB.hpp>
#include <vtkh/filters/Integrator.hpp>

#include <vtkm/worklet/MaskSelect.h>
#include <vtkm/worklet/particleadvection/ParticleAdvectionWorklets.h>
#include <vtkm/worklet/particleadvection/IntegratorStatus.h>


class ParticleAdvectWorkletB : public vtkm::worklet::WorkletMapField
{
public:
  vtkm::Id BatchSize;

  ParticleAdvectWorkletB() : BatchSize(-1) {}

  using ControlSignature = void(FieldIn idx,
                                ExecObject integrator,
                                ExecObject integralCurve,
                                FieldIn maxSteps,
                                FieldInOut active);
  using ExecutionSignature = void(_1 idx, _2 integrator, _3 integralCurve, _4 maxSteps, _5 active);
  using InputDomain = _1;
  using MaskType = vtkm::worklet::MaskSelect;

  template <typename IntegratorType, typename IntegralCurveType>
  VTKM_EXEC void operator()(const vtkm::Id& idx,
                            const IntegratorType* integrator,
                            IntegralCurveType& integralCurve,
                            const vtkm::Id& maxSteps,
                            vtkm::Id& active) const
  {
    vtkm::Particle particle = integralCurve.GetParticle(idx);

    vtkm::Vec3f inpos = particle.Pos;
    vtkm::FloatDefault time = particle.Time;
    bool tookAnySteps = false;

    //the integrator status needs to be more robust:
    // 1. you could have success AND at temporal boundary.
    // 2. could you have success AND at spatial?
    // 3. all three?

    vtkm::Id cnt = 0, canContinue = 1;
    integralCurve.PreStepUpdate(idx);
    do
    {
      //vtkm::Particle p = integralCurve.GetParticle(idx);
      //std::cout<<idx<<": "<<inpos<<" #"<<p.NumSteps<<" "<<p.Status<<" B= "<<BatchSize<<std::endl;
      vtkm::Vec3f outpos;
      auto status = integrator->Step(inpos, time, outpos);
      if (status.CheckOk())
      {
        integralCurve.StepUpdate(idx, time, outpos);
        tookAnySteps = true;
        inpos = outpos;
        cnt++;
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
          cnt++;

          //we took a step, so use this status to consider below.
          status = status2;
        }
      }

      integralCurve.StatusUpdate(idx, status, maxSteps);
      canContinue = integralCurve.CanContinue(idx);

    } while (canContinue == 1 && cnt < BatchSize);

    //Mark if any steps taken
    integralCurve.UpdateTookSteps(idx, tookAnySteps);
    active = canContinue;

    //particle = integralCurve.GetParticle(idx);
    //std::cout<<idx<<": "<<inpos<<" #"<<particle.NumSteps<<" "<<particle.Status<<std::endl;
  }
};

class DetermineCopy : public vtkm::worklet::WorkletMapField
{
public:
    using ControlSignature = void(FieldIn inActive, FieldIn inCopied, FieldOut doCopy);
    using ExecutionSignature = void(_1, _2, _3);
    using InputDomain = _1;

    VTKM_EXEC void operator() (const vtkm::Id& inActive,
                               const vtkm::Id& inCopied,
                               vtkm::Id& doCopy) const
    {
        doCopy = !inActive && !inCopied;
    }
};

class UpdateCopy : public vtkm::worklet::WorkletMapField
{
public:
    using ControlSignature = void(FieldIn in, FieldInOut out);
    using ExecutionSignature = void(_1, _2);
    using InputDomain = _1;

    VTKM_EXEC void operator() (const vtkm::Id& in,
                               vtkm::Id& out) const
    {
        if (in == 1)
          out = 1;
    }
};


int Integrator::Advect(std::vector<vtkh::Particle> &particles,
                       vtkm::Id MaxSteps,
                       std::vector<vtkh::Particle> &I,
                       std::vector<vtkh::Particle> &T,
                       std::vector<vtkh::Particle> &A,
                       std::vector<vtkm::worklet::ParticleAdvectionResult> *particleTraces,
                       vtkh::ThreadSafeContainer<vtkh::Particle, std::vector> &workerInactive,
                       vtkh::StatisticsDB& statsDB)
{
    vtkm::cont::ArrayHandle<vtkm::Particle> seedArray;
    int steps0 = SeedPrep(particles, seedArray);

#ifdef VTKH_USE_CUDA
    // This worklet needs some extra space on CUDA.
    vtkm::cont::cuda::ScopedCudaStackSize stack(16 * 1024);
    (void)stack;
#endif

    // Batch worklet...
    using ParticleType = vtkm::worklet::particleadvection::Particles;
    vtkm::Id numSeeds = static_cast<vtkm::Id>(seedArray.GetNumberOfValues());
    //Create and invoke the particle advection.
    vtkm::cont::ArrayHandleConstant<vtkm::Id> maxStepArr(MaxSteps, numSeeds);
    vtkm::cont::ArrayHandleIndex idxArray(numSeeds);

    ParticleType particlesObj(seedArray, MaxSteps);

    ParticleAdvectWorkletB worklet;
    worklet.BatchSize = batchSize;

    std::vector<vtkm::Id> activeVals(std::size_t(numSeeds), 1), copiedVals(std::size_t(numSeeds), 0);
    vtkm::cont::ArrayHandle<vtkm::Id> ActiveArr = vtkm::cont::make_ArrayHandle(activeVals);
    vtkm::cont::ArrayHandle<vtkm::Id> CopiedArr = vtkm::cont::make_ArrayHandle(copiedVals);
    vtkm::worklet::MaskSelect maskSelect(ActiveArr);

    vtkm::worklet::DispatcherMapField<ParticleAdvectWorkletB> dispatcher(worklet, maskSelect);

    vtkm::Particle p0 = seedArray.GetPortalControl().Get(0);

    bool done = false;
    vtkm::Id prevRemain = numSeeds;
    vtkm::Id loop = 0;
    int steps1 = 0;

    while (!done)
    {
        dispatcher.Invoke(idxArray, rk4, particlesObj, maxStepArr, ActiveArr);
        vtkm::Id currRemain = vtkm::cont::Algorithm::Reduce(ActiveArr, vtkm::Id(0));
        vtkm::Id justCompleted = prevRemain - currRemain;

        done = (currRemain == 0);

        TIMER_START("batchProcess");

        vtkm::cont::ArrayHandle<vtkm::Id> needCopy;
        DetermineCopy workletDC;
        vtkm::worklet::DispatcherMapField<DetermineCopy> dispatcherDC(workletDC);
        dispatcherDC.Invoke(ActiveArr, CopiedArr, needCopy);
        vtkm::Id numCopy = vtkm::cont::Algorithm::Reduce(needCopy, vtkm::Id(0));

        if (numCopy > 0)
        {
            vtkm::cont::ArrayHandle<vtkm::Particle> out;
            vtkm::cont::ArrayHandle<vtkm::Id> outIdx;
            vtkm::cont::Algorithm::CopyIf(seedArray, needCopy, out);
            vtkm::cont::Algorithm::CopyIf(idxArray, needCopy, outIdx);

            //update copied.
            UpdateCopy workletUC;
            vtkm::worklet::DispatcherMapField<UpdateCopy> dispatcherUC(workletUC);
            dispatcherUC.Invoke(needCopy, CopiedArr);
            std::vector<vtkh::Particle> tmp;
            tmp.resize(out.GetNumberOfValues());
            auto outP = out.GetPortalConstControl();
            auto outI = outIdx.GetPortalConstControl();

            std::vector<vtkh::Particle> tmpI, tmpT, tmpA;
            TIMER_START("batchCopyParticles");
            for (vtkm::Id i = 0; i < numCopy; i++)
            {
                vtkh::Particle p;
                p.p = outP.Get(i);
                p.blockIds = particles[outI.Get(i)].blockIds;
                steps1 += p.p.NumSteps;

                if (p.p.Status.CheckTerminate())
                    tmpT.push_back(p);
                else if (p.p.Status.CheckSpatialBounds())
                    tmpI.push_back(p);
                else if (p.p.Status.CheckOk())
                    tmpA.push_back(p);
                else
                    tmpT.push_back(p);
            }
            TIMER_STOP("batchCopyParticles");

            if (rank == 0) std::cout<<"Batch: Process: "<<out.GetNumberOfValues()<<" AsyncTIA: "<<tmpT.size()<<" "<<tmpI.size()<<" "<<tmpA.size()<<std::endl;
            if (!tmpI.empty())
                COUNTER_INC("batchSends", tmpI.size());

            workerInactive.Insert(tmpI);
            T.insert(T.end(), tmpT.begin(), tmpT.end());
            A.insert(A.end(), tmpA.begin(), tmpA.end());
        }

        if (rank == 0)
            std::cout<<loop<<": "<<prevRemain<<" --> "<<currRemain<<" : "<<justCompleted<<" done= "<<done<<std::endl;

        prevRemain = currRemain;
        loop++;

        TIMER_STOP("batchProcess");
    }

    COUNTER_INC("batchRounds", loop);

    vtkm::worklet::ParticleAdvectionResult result = vtkm::worklet::ParticleAdvectionResult(seedArray);

    if (particleTraces)
        particleTraces->push_back(result);

    vtkm::Particle p1 = seedArray.GetPortalControl().Get(0);
    if (rank == 0)
        std::cout<<p0.ID<<" "<<p0.Pos<<" ---> "<<p1.ID<<" "<<p1.Pos<<std::endl;

    int totalSteps = steps1-steps0;
    return totalSteps;
  }
