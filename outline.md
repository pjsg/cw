The problem is to build a wideband (say 100kHz) morse decoder in python to be used to monitor CW traffic on HF radios. This is split into two parts:

* Signal tracker
* Morse Decoder


The signal tracker consumes a WAV stream (say 250ksps) which contains (roughly) 100kHz of decoded RF. There will be multiple morse transmissions on different frequencies at the same time. The code should find the location of the signals (e.g. by using an FFT to convert to a spectrum and then picking out the peaks). By using a polyphase filter bank, each morse signal can be extracted efficiently. Then the energy in each signal should be determined, pulses identified and output in JSON lines format. Each JSON object should contain these fields:

* timestamp: Timestamp of start of pulse (in seconds since start of the stream)
* width: Width of pulse (in seconds)
* frequency: Center frequency of pulse (in Hz)

Note that detecting pulses may require tracking the amplitude of signals for a specific frequency. However, be careful around distinguishing real signals from random noise.
If the data is processed in timeslices (e.g. 5 second chunks), then care needs to be taken to ensure that pulses that cross chunk boundaries are handled correctly. It may be best to process overlapping chunks in time and then eliminate duplicates as a post processing step.

The code should be able to handle a wide range of sample rates from 8ksps up to 500ksps.

A couple of sample files are sample.wav and test1.wav

The morse decoder should consume the JSON objects output by the signal tracker and decode them into ascii text. It should generate output in JSON lines format. Each JSON object should contain the following fields:

* timestamp: Timestamp of the start of the first decoded symbol
* text: One or more decoded characters
* frequency: Center frequency of the input pulses

This will involve grouping the pulses by frequency, estimating the morse parameters and then finally decoding the signal. The morse parameters are:

* Dit time
* Dah time
* Intra symbol space
* Inter symbol space
* Inter word space

Machine generated morse adopts the standard ratios, but human generated morse often has different ratios and there is variability from symbol to symbol.
If the data is processed in timeslices (e.g. 5 second chunks), then care needs to be taken to ensure that symbols that cross chunk boundaries are handled correctly. It may be best to process overlapping chunks in time and then eliminate duplicates as a post processing step.
