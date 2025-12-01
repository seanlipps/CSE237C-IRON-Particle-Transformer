#include <adf.h>
#include "include.h"
#include <vector>

using namespace adf;

class simpleGraph : public adf::graph {
public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  kernel resadd_1 [1];

  simpleGraph(){

    AIE_IN = input_plio::create("DataIn", plio_128_bits, "data/input.txt", 1280);
    AIE_OUT = output_plio::create("DataOut", plio_128_bits, "data/out_sim.txt", 1280);

    #include "layer_graph.h"

  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  adf::event::handle h_latency =
    adf::event::start_profiling(mygraph.AIE_IN, mygraph.AIE_OUT,
                                adf::event::io_stream_start_difference_cycles);
  mygraph.run(ITERATIONS);
  mygraph.end();
  long long latency_cycles = adf::event::read_profiling(h_latency);
  adf::event::stop_profiling(h_latency);
  const int AIE_clock_Hz = 1200000000;
  printf("\n\n\n--------GRAPH LATENCY    (First in  -> First out) : %lld cycles, %.1f ns\n\n\n",
         latency_cycles, (1e9 * (double)latency_cycles) / AIE_clock_Hz);
  return 0;
}
