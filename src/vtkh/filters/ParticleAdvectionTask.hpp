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

        //KB changes
        ADD_TIMER("CPUworker_sleep");
        ADD_TIMER("GPUworker_sleep");
        ADD_COUNTER("CPUworker_naps");
        ADD_COUNTER("GPUworker_naps");
        ADD_COUNTER("GPU");
        ADD_COUNTER("CPU");
    }
    ~ParticleAdvectionTask()
    {
#ifdef VTKH_STD_THREAD
      for (auto &w : workerThreads)
        w.join();
#endif
    }

    void Init(const std::list<Particle> &particles, int N, int _sleepUS)
    {
        numWorkerThreads = 1;
        TotalNumParticles = N;
        sleepUS = _sleepUS;
        
        //KB change
        if(0) { // (TNumParts && TMaxSteps) || (TSeedMeth && TNumRanks && TStepSize)) {
          activeG.Assign(particles); //what about assigning to GPU? May need another "Oracle"-like thing here
          DBG("Oracle started with GPU"<<std::endl);
        } else {
          activeC.Assign(particles); //maybe all particles should start in one active queue
          DBG("Oracle started with CPU instead"<<std::endl);
        }

        inactive.Clear();
        terminated.Clear();
    }

    //KB change
    void OracleInit (float PAstepsize, int PAseedmethod, int PAmaxsteps, int PAnumseeds)
    {
        if(PAstepsize > .05) TStepSize = 1;
        else TStepSize = 0;

        if(PAseedmethod == 0 || PAseedmethod == 1) TSeedMeth = 1;
        else TSeedMeth = 0;

        if(PAmaxsteps > 1000) TMaxSteps = 1;
        else TMaxSteps = 0;

        if(m_NumRanks > 10) TNumRanks == 1;
        else TNumRanks == 0;

        if(PAnumseeds > 1000) TNumParts == 1;
        else TNumParts == 0;
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

#ifndef VTKH_STD_THREAD
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
                this->cpuWork(); //KB change
            }
        }
#else
        workerThreads.push_back(std::thread(ParticleAdvectionTask::cpuWorker, this));
        workerThreads.push_back(std::thread(ParticleAdvectionTask::gpuWorker, this)); //KB change
        this->Manage();
#endif
    }

#ifdef VTKH_STD_THREAD //KB change
    static void cpuWorker(ParticleAdvectionTask *t)
    {
      std::cerr<<"Created CPU thread"<<std::endl;
      t->cpuWork();
    }

    static void gpuWorker(ParticleAdvectionTask *t)
    {
      std::cerr<<"Created GPU thread"<<std::endl;
      t->gpuWork();
    }
#endif
 
    //KB changes
    void cpuWork()
    {
      std::vector<ResultT> tracesC;

        while (!CheckDone())
        {
            std::vector<Particle> particlesC;
            if (activeC.Get(particlesC))
            {
                std::list<Particle> I, T, A;
                
                COUNTER_INC("CPU", 1);

                DataBlockIntegrator *blkC = filter->GetBlock(particlesC[0].blockIds[0]);

                TIMER_START("advectC");
                WDBG("CPU WORKER: Integrate "<<particlesC<<" --> "<<std::endl);
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

    //KB changes
    void gpuWork()
    {
      std::vector<ResultT> tracesG;

        while (!CheckDone())
        {
            std::vector<Particle> particlesG;
            if (activeG.Get(particlesG))
            {
                std::list<Particle> I, T, A;
                
                COUNTER_INC("GPU", 1);

                //DataBlockIntegrator *blkG = filter->GetBlock(particlesG[0].blockIds[0]);
                DataBlockIntegrator *blkG = filter->GetBlock(m_Rank + 1000);

                TIMER_START("advectG");
                WDBG("GPU WORKER: Integrate "<<particlesG<<" --> "<<std::endl);
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
    }

    //KB changes
    void Manage()
    {
        DBG("manage_bm: "<<boundsMap<<std::endl);

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
             if(in.size() > 0) { //kind of acts like an Oracle, add other conditions later
               DBG("Adding to CPU"<<std::endl);
                 activeC.Insert(in);
               } else {
                 DBG("Adding to GPU<<std::endl");
                 activeG.Insert(in);
               }

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

#ifdef VTKH_STD_THREAD
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
    bool TStepSize, TMaxSteps, TSeedMeth, TNumParts, TNumRanks;

    bool done, begin;
    vtkh::Mutex stateLock;
    BoundsMap boundsMap;
    ParticleAdvection *filter;
};
}

#endif //VTK_H_PARTICLE_ADVECTION_TASK_HPP
