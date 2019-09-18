#ifndef VTK_H_PARTICLE_ADVECTION_TASK_HPP
#define VTK_H_PARTICLE_ADVECTION_TASK_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>
#include <vtkh/utils/ThreadSafeContainer.hpp>
#include <vtkh/filters/ParticleAdvection.hpp>
#include <vtkh/filters/communication/BoundsMap.hpp>

#ifdef ENABLE_LOGGING
#define DBG(msg) vtkh::Logger::GetInstance("out")->GetStream()<<msg
#define WDBG(msg) vtkh::Logger::GetInstance("wout")->GetStream()<<msg
#else
#define DBG(msg)
#define WDBG(msg)
#endif

#define ORACLE1

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
        sleepUS(100)
    {
        m_Rank = vtkh::GetMPIRank();
        m_NumRanks = vtkh::GetMPISize();
        communicator.RegisterMessages(2, std::min(64, m_NumRanks-1), 128, std::min(64, m_NumRanks-1));
        int nDoms = pa->GetInput()->GetNumberOfDomains();
        for (int i = 0; i < nDoms; i++)
        {
            vtkm::Id id;
            vtkm::cont::DataSet ds;
            pa->GetInput()->GetDomain(i, ds, id);
            communicator.AddLocator(id, ds);
        }

        ADD_TIMER("CPUworker_sleep");
        ADD_TIMER("GPUworker_sleep");
        ADD_COUNTER("CPUworker_naps");
        ADD_COUNTER("GPUworker_naps");
        ADD_COUNTER("GPU");
        ADD_COUNTER("CPU");
    }
    ~ParticleAdvectionTask()
    {
#ifndef VTKH_USE_OPENMP
      for (auto &w : workerThreads)
        w.join();
#endif
    }

    void Init(const std::list<Particle> &particles, int N, int _sleepUS, int steps)
    {
        numWorkerThreads = 1;
        TotalNumParticles = N;
        sleepUS = _sleepUS;
        int everyN = N / 8;        

        //initialize oracle
        if(N <= 1000) {
          activeC.Assign(particles); 
          DBG("Oracle started with CPU"<<std::endl);
        } 
        else if (N >= 4000) { //Note: this problem size with "tiny tiny" duration is a CPU win
          activeG.Assign(particles);
          useGPU = 1; //tell manage thread that this rank started on GPU
          DBG("Oracle started with GPU instead"<<std::endl);
        }
        else { 
          hybridGPU = 1; //tell manage thread that this rank started in hybrid range
#ifndef ORACLE3          
          if (steps <= 250) { //initialize oracle1 or oracle2
            activeC.Assign(particles);
            DBG("Oracle went to hybrid range and started with CPU");
          } else {
            activeG.Assign(particles);
            DBG("Oracle went to hybrid range and started with GPU");
          }
#else
          //initialize oracle3
          std::list<Particle> gpuList;
          std::list<Particle> cpuList;
          int i = 0;
          for(auto it = particles.cbegin(); it != particles.cend(); it++) {
            if(i % 10 == 0) {
              cpuList.push_front(*it);
            }
            else {
              gpuList.push_front(*it);
            }
            i++;
          }
          activeC.Assign(cpuList);
          activeG.Assign(gpuList);
          cpuList.clear(); gpuList.clear();
#endif        
        }

        inactive.Clear();
        terminated.Clear();
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
        DBG("Go_bm: "<<boundsMap<<std::endl);
        DBG("actives= "<<activeC<<activeG<<std::endl);

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
	for (auto &w : workerThreads)  w.join();
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

      if(device_tracker.CanRunOn(vtkm::cont::DeviceAdapterTagSerial())) {
        device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagSerial());
	WDBG("CPU thread forcing vtkm to run with Device Adapter Tag Serial"<<std::endl);
      }
      else
      {
        std::stringstream msg;
        msg << "CPU thread Failed to set up Device Tag Serial " << std::endl;
        throw Error(msg.str());
      }

        while (!CheckDone())
        {
            std::vector<Particle> particlesC;
            if (activeC.Get(particlesC))
            {
                std::list<Particle> I, T, A;

                workingOnC = particlesC.size();
                COUNTER_INC("CPU", 1);

                DataBlockIntegrator *blkC = filter->GetBlock(particlesC[0].blockIds[0]);

                WDBG("CPU WORKER: Integrate "<<particlesC<<" --> "<<std::endl);
                TIMER_START("advectC");
                int n = filter->InternalIntegrate<ResultT>(*blkC, particlesC, I, T, A, tracesC);
                TIMER_STOP("advectC");
                COUNTER_INC("advectStepsC", n);
                WDBG("CPU TIA: "<<T<<" "<<I<<" "<<A<<std::endl<<std::endl);

                worker_terminated.Insert(T);
                worker_active.Insert(A);
                worker_inactive.Insert(I);
            }
            else
            {
                TIMER_START("CPUworker_sleep");
                usleep(sleepUS);
                TIMER_STOP("CPUworker_sleep");
                COUNTER_INC("CPUworker_naps", 1);
            }
        }
        WDBG("CPU WORKER is DONE"<<std::endl);
        results.Insert(tracesC);
    }

    void gpuWork()
    {
      std::vector<ResultT> tracesG;
      vtkm::cont::RuntimeDeviceTracker &device_tracker
                                       = vtkm::cont::GetRuntimeDeviceTracker();

      if(device_tracker.CanRunOn(vtkm::cont::DeviceAdapterTagCuda())) {
        device_tracker.ForceDevice(vtkm::cont::DeviceAdapterTagCuda());
	WDBG("Forcing GPU thread to run vtkm with Device Adapter Tag CUDA"<<std::endl);
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
                std::list<Particle> I, T, A;
                
                workingOnG = particlesG.size();
                COUNTER_INC("GPU", 1);

                DataBlockIntegrator *blkG = filter->GetBlock(m_Rank + 1000);

                WDBG("GPU WORKER: Integrate "<<particlesG<<" --> "<<std::endl);
                TIMER_START("advectG");
                int n = filter->InternalIntegrate<ResultT>(*blkG, particlesG, I, T, A, tracesG);
                TIMER_STOP("advectG");
                COUNTER_INC("advectStepsG", n);
                WDBG("GPU TIA: "<<T<<" "<<I<<" "<<A<<std::endl<<std::endl);

                worker_terminated.Insert(T);
                worker_active.Insert(A);
                worker_inactive.Insert(I);
            }
            else
            {
                TIMER_START("GPUworker_sleep");
                usleep(sleepUS);
                TIMER_STOP("GPUworker_sleep");
                COUNTER_INC("GPUworker_naps", 1);
            }
        }
        WDBG("GPU WORKER is DONE"<<std::endl);
        results.Insert(tracesG);
	WDBG("RESULTS size "<<results.Size()<<std::endl);
    }

    void Manage()
    {
        DBG("manage_bm: "<<boundsMap<<std::endl);
        
        int almostDone = (TotalNumParticles * .87);
        int N = 0;

        DBG("Begin TIA: "<<terminated<<" "<<inactive<<" "<<activeC<<" "<<activeG<<std::endl);
        MPI_Comm mpiComm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

        while (true)
        {
            DBG("MANAGE TIA: "<<terminated<<" "<<worker_inactive<<" "<<activeC<<" "<<activeG<<std::endl<<std::endl);
            std::list<Particle> out, in, term;
            worker_inactive.Get(out);
            worker_terminated.Get(term);

            int numTermMessages;
            communicator.Exchange(out, in, term, numTermMessages);
            int numTerm = term.size() + numTermMessages;

            if (!in.empty()) {
              if(useGPU) { //maybe add special cases here later
                activeG.Insert(in);
                DBG("Adding to GPU"<<std::endl);
              } else if (hybridGPU) { 
#ifdef ORACLE1      
                if(workingOnG + activeG.Size() + in.size() > 100) { // if GPU still has lots to do, continue with GPU
                  DBG("Adding to hybrid GPU"<<std::endl);
                  activeG.Insert(in);
                }
                else { //GPU work is small, use CPU to finish up, maybe use workingOnC later
                  DBG("Adding to hybrid CPU"<<std::endl);
                  activeC.Insert(in);
                }
#elif ORACLE2
                if (N < almostDone) { //if we aren't almost done, add to GPU
                  DBG("Adding to hybrid GPU"<<std::endl);
                  activeG.Insert(in);
                }
                else { //we're almost done so start up the CPU
                  DBG("Adding to hybrid CPU"<<std::endl);
                  activeC.Insert(in);
                }
#else
                //ORACLE3
                activeC.Insert(in);
#endif
             } else {
               DBG("Adding to CPU"<<std::endl);
               activeC.Insert(in);
             }
             
             /* todo: Implementation for Oracle3
             else if (hybridGPU) {    
               DBG("Adding to hybrid CPU"<<std::endl);
               std::list<Particle> gpuList;
               std::list<Particle> cpuList;
               int i = 0;
               for(auto it = in.cbegin(); it != in.cend(); it++) {
                 if(i % 8 == 0) {
                   cpuList.push_front(*it);
                 }
                 else {
                   gpuList.push_front(*it);
                 }
                 i++;
               }
               activeC.Assign(cpuList);
               activeG.Assign(gpuList);
               cpuList.clear(); gpuList.clear();
             } 
             */

               DBG("ActivesC: "<<activeC<<std::endl<<"ActiveG: "<<activeG<<std::endl);

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
        DBG("TIA: "<<terminated<<" "<<inactive<<" "<<activeC<<" "<<activeG<<" WI= "<<worker_inactive<<std::endl);
        DBG("RESULTS= "<<results.Size()<<std::endl);
        DBG("DONE_"<<m_Rank<<" "<<terminated<<" "<<activeC<<" "<<activeG<<" "<<inactive<<std::endl);
        SetDone();
    }

    int m_Rank, m_NumRanks;
    int TotalNumParticles;

#ifndef VTKH_USE_OPENMP
    std::vector<std::thread> workerThreads;
#endif

    using ParticleList = vtkh::ThreadSafeContainer<Particle, std::list>;
    using ResultsVec = vtkh::ThreadSafeContainer<ResultT, std::vector>;

    ParticleMessenger communicator;
    ParticleList activeC, activeG, inactive, terminated;
    ParticleList worker_active, worker_inactive, worker_terminated;
    ResultsVec results;

    int numWorkerThreads;
    int sleepUS;
    int useGPU = 0, hybridGPU = 0;
    int workingOnC = 0, workingOnG = 0;

    bool done, begin;
    vtkh::Mutex stateLock;
    BoundsMap boundsMap;
    ParticleAdvection *filter;
};
}

#endif //VTK_H_PARTICLE_ADVECTION_TASK_HPP
