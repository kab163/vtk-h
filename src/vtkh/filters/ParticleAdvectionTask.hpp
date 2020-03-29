#ifndef VTK_H_PARTICLE_ADVECTION_TASK_HPP
#define VTK_H_PARTICLE_ADVECTION_TASK_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>
#include <vtkh/utils/ThreadSafeContainer.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/communication/BoundsMap.hpp>

//#ifdef VTKH_ENABLE_LOGGING
//#define DBG(msg) vtkh::Logger::GetInstance("out")->GetStream()<<msg
//#define WDBG(msg) vtkh::Logger::GetInstance("wout")->GetStream()<<msg
//#else
//#define DBG(msg)
//#define WDBG(msg)
//#endif

#define THRESHOLD 125
#define FACTOR .95

#define CPUONLY

namespace vtkh
{
template <typename ResultT>
class ParticleAdvectionTask
{
public:
    ParticleAdvectionTask(MPI_Comm comm, const vtkh::BoundsMap &bmap, ParticleAdvection *pa) :
        numWorkerThreads(-1),
        done(false),
        begin(false),
        communicator(comm, bmap),
        boundsMap(bmap),
        filter(pa),
        sleepUS(100),
	batchSize(-1)
    {
        m_Rank = vtkh::GetMPIRank();
        m_NumRanks = vtkh::GetMPISize();
        communicator.RegisterMessages(2, std::min(64, m_NumRanks-1), 128, std::min(64, m_NumRanks-1));
        //ADD_TIMER("worker_sleep");
        //ADD_COUNTER("worker_naps");
        // Adding Events
        //stats.AddEvent("CPU_Advect");
        //stats.AddEvent("GPU_Advect");
        //stats.AddEvent("CPU_Sleep");
        //stats.AddEvent("GPU_Sleep");

        ADD_TIMER("CPUworker_sleep");
        ADD_TIMER("GPUworker_sleep");
        ADD_COUNTER("CPUworker_naps");
        ADD_COUNTER("GPUworker_naps");
        ADD_COUNTER("GPU");
        ADD_COUNTER("CPU");
    }
    ~ParticleAdvectionTask()
    {
    }

    void Init(const std::vector<Particle> &particles, int N, int _sleepUS, int _batchSize)
    {
        numWorkerThreads = 1;
        TotalNumParticles = N;
        sleepUS = _sleepUS;
        almostDone = 60000; //(N * .95);
	batchSize = _batchSize;

        if (OracleDecidedToUseGPU((N/m_NumRanks))) {
          activeG.Assign(particles);
          //DBG("Oracle started with GPU"<<std::endl);
        }
        else {
          activeC.Assign(particles);
          //DBG("Oracle started with CPU instead"<<std::endl);
        }

        inactive.Clear();
        terminated.Clear();
    }

    int OracleDecidedToUseGPU(int n)
    {
#ifdef ONLYGPU
      return 1; //run on GPU only
#endif
#ifdef CPUONLY
      return 0; //run on CPU only
#endif
#ifdef ORACLE1
      if(n <= (THRESHOLD))
        return 0; //run on CPU
      else
        return 1; //run on GPU
#endif
      return 0; //shouldn't get to this point...
    }

    bool CheckDone()
    {
        bool val;
        stateLock.Lock();
        val = done;
        stateLock.Unlock();
        return val;
    }
    void SetDone()
    {
        stateLock.Lock();
        done = true;
        stateLock.Unlock();
    }

    bool GetBegin()
    {
        bool val;
        stateLock.Lock();
        val = begin;
        stateLock.Unlock();
        return val;
    }

    void SetBegin()
    {
        stateLock.Lock();
        begin = true;
        stateLock.Unlock();
    }

    void Go()
    {
        //DBG("Go_bm: "<<boundsMap<<std::endl);
        //DBG("actives " << std::endl
          // << "CPU : " << activeC << std::endl
          // << "GPU : " << activeG << std::endl);

#ifdef VTKH_USE_OPENMP
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            #pragma omp parallel num_threads(1)
            {
                this->Manage();
            }

            #pragma omp section
            #pragma omp parallel num_threads(numWorkerThreads)
            #pragma omp master
            {
                this->cpuWork();
            }
        }
#else
        workerThreads.push_back(std::thread(ParticleAdvectionTask::cpuWorker, this));
        workerThreads.push_back(std::thread(ParticleAdvectionTask::gpuWorker, this));
        this->Manage();
        for (auto &t : workerThreads)
            t.join();
#endif
    }

#ifndef VTKH_USE_OPENMP
    static void cpuWorker(ParticleAdvectionTask *t)
    {
      t->cpuWork();
    }

    static void gpuWorker(ParticleAdvectionTask *t)
    {
      t->gpuWork();
    }
#endif

    void cpuWork()
    {
      std::vector<ResultT> tracesC;
      vtkm::cont::RuntimeDeviceTracker &device_tracker
                                       = vtkm::cont::GetRuntimeDeviceTracker();

      if(device_tracker.CanRunOn(vtkm::cont::DeviceAdapterTagOpenMP())) {
        device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagOpenMP());
      }
      else
      {
        std::stringstream msg;
        msg << "CPU thread Failed to set up Device Tag OpenMP " << std::endl;
        throw Error(msg.str());
      }

        while (!CheckDone())
        {
            std::vector<Particle> particlesC;
            if (activeC.Get(particlesC))
            {
                std::vector<Particle> I, T, A;

                workingOnC = particlesC.size();
                COUNTER_INC("CPU", 1);

                DataBlockIntegrator *blkC = filter->GetBlock(particlesC[0].blockIds[0]);

                //stats.Begin("CPU_Advect");
                TIMER_START("advectC");
                int n = filter->InternalIntegrate<ResultT>(*blkC, particlesC, I, T, A, tracesC);
                TIMER_STOP("advectC");
                //stats.End("CPU_Advect");
                COUNTER_INC("advectStepsC", n);

                worker_terminated.Insert(T);
                worker_active.Insert(A);
                worker_inactive.Insert(I);
            }
            else
            {
                //stats.Begin("CPU_Sleep");
                TIMER_START("CPUworker_sleep");
                usleep(sleepUS);
                TIMER_STOP("CPUworker_sleep");
                //stats.End("CPU_Sleep");
                COUNTER_INC("CPUworker_naps", 1);
            }
        }
        results.Insert(tracesC);
    }

    void gpuWork()
    {
      std::vector<ResultT> tracesG;
      vtkm::cont::RuntimeDeviceTracker &device_tracker
                                       = vtkm::cont::GetRuntimeDeviceTracker();

      if(device_tracker.CanRunOn(vtkm::cont::DeviceAdapterTagCuda())) {
        device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagCuda());
      }
      else
      {
        std::stringstream msg;
        msg << "GPU thread Failed to set up Device Adapter Tag CUDA " << std::endl;
        throw Error(msg.str());
      }

        while (!CheckDone())
        {
            std::vector<Particle> particlesG;
            if (activeG.Get(particlesG))
            {
                std::vector<Particle> I, T, A;

                workingOnG = particlesG.size();
                COUNTER_INC("GPU", 1);

                DataBlockIntegrator *blkG = filter->GetBlock(m_Rank + 1000);

                //stats.Begin("GPU_Advect");
                TIMER_START("advectG");
                int n = filter->InternalIntegrate<ResultT>(*blkG, particlesG, I, T, A, tracesG);
                TIMER_STOP("advectG");
                //stats.End("GPU_Advect");
                COUNTER_INC("advectStepsG", n);

                worker_terminated.Insert(T);
                worker_active.Insert(A);
                worker_inactive.Insert(I);
            }
            else
            {
                //stats.Begin("GPU_Sleep");
                TIMER_START("GPUworker_sleep");
                usleep(sleepUS);
                TIMER_STOP("GPUworker_sleep");
                //stats.End("GPU_Sleep");
                COUNTER_INC("GPUworker_naps", 1);
            }
        }
        results.Insert(tracesG);
    }

    void Manage()
    {
        int N = 0;
        MPI_Comm mpiComm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

        while (true)
        {
            std::vector<Particle> out, in, term;
            worker_inactive.Get(out);
            worker_terminated.Get(term);

            int numTermMessages;
            communicator.Exchange(out, in, term, numTermMessages);
            int numTerm = term.size() + numTermMessages;

            if (!in.empty()) {
              if (OracleDecidedToUseGPU((workingOnG + activeG.Size() + in.size()))) {
                activeG.Insert(in);
              }
              else {
                activeC.Insert(in);
              }
              /* At some point, I can Get a certain number of elements of the TSC to put into the active queues */
            }
            if (!term.empty())
                terminated.Insert(term);

            N += numTerm;
            if (N > TotalNumParticles)
                throw "Particle count error";
            if (N == TotalNumParticles)
                break;

            if (activeC.Empty() && activeG.Empty())
            {//Could eventually put the advect particle code here
                TIMER_START("sleep");
                usleep(sleepUS);
                TIMER_STOP("sleep");
                COUNTER_INC("naps", 1);
                communicator.CheckPendingSendRequests();
            }
        }
        SetDone();
    }

    int m_Rank, m_NumRanks;
    int TotalNumParticles;

#ifndef VTKH_USE_OPENMP
    std::vector<std::thread> workerThreads;
#endif

    using ParticleList = vtkh::ThreadSafeContainer<Particle, std::vector>;
    using ResultsVec = vtkh::ThreadSafeContainer<ResultT, std::vector>;

    ParticleMessenger communicator;
    ParticleList activeC, activeG, inactive, terminated;
    ParticleList worker_active, worker_inactive, worker_terminated;
    ResultsVec results;

    int numWorkerThreads;
    int sleepUS;
    int batchSize;
    int numSteps = 0;
    int workingOnC = 0, workingOnG = 0;
    int almostDone = 0;

    bool done, begin;
    vtkh::Mutex stateLock;
    BoundsMap boundsMap;
    ParticleAdvection *filter;
};
}

#endif //VTK_H_PARTICLE_ADVECTION_TASK_HPP

