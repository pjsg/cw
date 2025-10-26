# Batch vs Streaming Comparison

## Current Status

**The batch and streaming versions do NOT produce identical results.**

### Test Results (sample.wav at 750 Hz)

| Version | Output | Status |
|---------|--------|--------|
| **Batch** | "N4LSJ CA" | ✓ Correct |
| **Streaming** | "I J A" | ✗ Partial (missing N, 4, L, S) |

---

## Why They Differ

### Design Differences

**Batch Mode:**
- Processes complete file at once
- Can analyze all gaps globally
- Estimates parameters from all 24 pulses
- Splits characters using gap analysis across entire dataset
- More accurate for pre-recorded files

**Streaming Mode:**
- Processes pulses incrementally as they arrive
- Must estimate parameters from first 10 pulses only
- Uses adaptive gap thresholds based on limited data
- Designed for real-time operation where timeouts matter
- Optimized for live audio streams

### Technical Issues

1. **Parameter Estimation:**
   - Batch: Uses all pulses for better statistics
   - Streaming: Limited to first 10 pulses, less accurate

2. **Gap Detection:**
   - Batch: Can look ahead/behind for context
   - Streaming: Can only use consecutive pulses

3. **Word Boundaries:**
   - Batch: Post-processes all pulses together
   - Streaming: Must decide boundaries immediately

4. **Timeout Handling:**
   - Batch: No timeouts needed (all data available)
   - Streaming: Uses pulse timestamps for timeouts (not wall-clock)

---

## When To Use Which

### Use **Batch Mode** for:
- ✓ Post-processing recorded files
- ✓ Maximum accuracy needed
- ✓ Complete transmissions available
- ✓ Testing and development
- ✓ File analysis and logging

### Use **Streaming Mode** for:
- ✓ Live audio monitoring
- ✓ Real-time decoding
- ✓ Continuous operation (24/7)
- ✓ SDR integration
- ✓ Low-latency applications

---

## Recommendations

### For Pre-Recorded Files
**Use the batch version** - it's more accurate and designed for this use case:
```bash
python signal_tracker.py file.wav | python morse_decoder.py /dev/stdin
```

### For Live Audio
**Use the streaming version** - it handles infinite streams properly:
```bash
arecord -f S16_LE -r 12000 -c 1 -t raw | \
  python signal_tracker_streaming.py -r 12000 | \
  python morse_decoder_streaming.py
```

---

## Future Work

To improve streaming accuracy:

1. **Increase min_pulses** - Estimate from 20+ pulses instead of 10
2. **Adaptive learning** - Update parameters as more pulses arrive
3. **Lookahead buffer** - Buffer a few pulses before processing
4. **Better gap clustering** - Use online clustering algorithms
5. **Hybrid approach** - Use batch algorithm with sliding window

---

## Conclusion

Both versions are functional but optimized for different use cases:

- **Batch**: Best accuracy for files
- **Streaming**: Required for real-time operation

For production systems processing live audio, **streaming is essential** despite slightly lower accuracy. The trade-off is acceptable for real-time monitoring applications.

For maximum accuracy on recorded files, **use batch mode**.
