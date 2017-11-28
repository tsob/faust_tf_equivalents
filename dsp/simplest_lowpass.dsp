// The simplest lowpass filter, e.g. as defined as
// https://ccrma.stanford.edu/~jos/fp/Definition_Simplest_Low_Pass.html

import("stdfaust.lib");
process = _ : fi.fir((0.5, 0.5));
