#include <vtkh/vtkh.hpp>
#include <vtkh/StatisticsDB.hpp>

namespace vtkh
{
StatisticsDB stats;

void
StatisticsDB::DumpStats(const std::string &fname, const std::string &preamble, bool append)
{
    stats.calcStats();

#ifdef VTKH_PARALLEL
    int rank = vtkh::GetMPIRank();
    if (rank != 0)
        return;
#endif

    if (!append || !outputStream.is_open())
        outputStream.open(fname, std::ofstream::out);

    if (!preamble.empty())
        outputStream<<preamble;

    if (!stats.timers.empty())
    {
        outputStream<<"TIMERS:"<<std::endl;
        for (auto &ti : stats.timers)
            outputStream<<ti.first<<": "<<ti.second.GetTime()<<std::endl;
        outputStream<<std::endl;
        outputStream<<"TIMER_STATS"<<std::endl;
        for (auto &ti : stats.timers)
            outputStream<<ti.first<<" "<<stats.timerStat(ti.first)<<std::endl;
    }
    if (!stats.counters.empty())
    {
        outputStream<<std::endl;
        outputStream<<"COUNTERS:"<<std::endl;
        for (auto &ci : stats.counters)
            outputStream<<ci.first<<" "<<stats.totalVal(ci.first)<<std::endl;
        outputStream<<std::endl;
        outputStream<<"COUNTER_STATS"<<std::endl;
        for (auto &ci : stats.counters)
            outputStream<<ci.first<<" "<<stats.counterStat(ci.first)<<std::endl;
    }
/*    if (!stats.events.empty())
    {
        outputStream<<std::endl;
        outputStream<<"EVENT_STATS:"<<std::endl;
        for (auto &ei : stats.events)
            outputStream<<ei.first<<" "<<stats.eventStat(ei.first)<<std::endl;
    }*/
}

void StatisticsDB::DumpEvents(const std::string &eventFileName)
{
    
#ifdef VTKH_PARALLEL
    int nRanks = vtkh::GetMPISize();
    int rank = vtkh::GetMPIRank();
    if (rank != 0)
        return;
#else
    int nRanks = 1;
    int rank = 0;
#endif

    std::ofstream outE;
    outE.open(eventFileName);    

    double diff = 0;
    int loopCount = 0;
    for (int i = 0; i < nRanks; i++)
    {
        loopCount = 0;
        for (auto it = stats.eventStats[i].begin(); it != stats.eventStats[i].end(); it++)
        {
            for (auto h = stats.eventStats[i][it->first].history.begin();
                 h != stats.eventStats[i][it->first].history.end(); h++)
            {
                diff = h->second - h->first;
		if(diff < 0.000005 && it->first == "communication") ;
                else   
		{
		    if(diff < 0.00000 && it->first == "integration") h->second = h->first + 0.00005;
	            
                    outE<<i<<","<<it->first<<"_"<<i<<","<<h->first<<","<<h->second<<","<<it->first<<std::endl;
                }
                loopCount++;
            }
        }
    }
}

};
