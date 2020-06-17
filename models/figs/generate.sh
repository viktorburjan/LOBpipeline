 
convert                                                  \
  -delay 15                                              \
   $(for i in $(seq 0 1 199); do echo snapshot-${i}.png; done) \
  -loop 0                                                \
   animated2.gif
