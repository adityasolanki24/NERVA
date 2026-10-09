# Pivot attempt aborted before generation

Protocol `f8728dc`, implementation `3dca496`. Both workers fail in the installed
planner geometry verifier due to the unsupported Eigen transform return type
of Placo 0.6.3's Footstep.frame binding. No raw motions or fits were produced.
Elapsed 3.88 s. Preserve this abort separately from the polygon-verifier retry.
