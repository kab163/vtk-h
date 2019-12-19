#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>

namespace vtkh
{
StatisticsDB statsDB;

void
StatisticsDB::DumpStats(const std::string &fname, const std::string &preamble, bool append)
{
    statsDB.calcStats();

#ifdef VTKH_PARALLEL
    int rank = vtkh::GetMPIRank();
    if (rank != 0)
        return;
#endif

    if (!append || !outputStream.is_open())
        outputStream.open(fname, std::ofstream::out);

    if (!preamble.empty())
        outputStream<<preamble;

    if (!statsDB.timers.empty())
    {
        outputStream<<"TIMERS:"<<std::endl;
        for (auto &ti : statsDB.timers)
            outputStream<<ti.first<<": "<<ti.second.GetTime()<<std::endl;
        outputStream<<std::endl;
        outputStream<<"TIMER_STATS"<<std::endl;
        for (auto &ti : statsDB.timers)
            outputStream<<ti.first<<" "<<statsDB.timerStat(ti.first)<<std::endl;
    }
    if (!statsDB.counters.empty())
    {
        outputStream<<std::endl;
        outputStream<<"COUNTERS:"<<std::endl;
        for (auto &ci : statsDB.counters)
            outputStream<<ci.first<<" "<<statsDB.totalVal(ci.first)<<std::endl;
        outputStream<<std::endl;
        outputStream<<"COUNTER_STATS"<<std::endl;
        for (auto &ci : statsDB.counters)
            outputStream<<ci.first<<" "<<statsDB.counterStat(ci.first)<<std::endl;
    }
}

};
