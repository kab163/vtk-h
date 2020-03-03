#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>

namespace vtkh
{
StatisticsDB statsDB;

void
StatisticsDB::DumpStats(const std::string &fname, const std::string &preamble, bool append)
{
    statsDB.calcStats();

    int numRanks = 1;
#ifdef VTKH_PARALLEL
    int rank = vtkh::GetMPIRank();
    numRanks = vtkh::GetMPISize();
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
    if (!statsDB.eventStats.empty())
    {
        std::string eventFname = fname + ".events";
        std::ofstream eventStream;
        eventStream.open(eventFname, std::ofstream::out);

        for (int i = 0; i < numRanks; i++)
        {
            auto events = eventStats[i];
            for (auto ei : events)
            {
                auto eventNm = ei.first;
                auto eventHistory = ei.second;
                eventStream<<eventNm<<"_"<<i<<",";
                for (auto &hi : eventHistory.history)
                    eventStream<<hi.first<<","<<hi.second<<",";
                eventStream<<eventNm<<std::endl;
            }
        }
        eventStream.close();
    }
}

};
