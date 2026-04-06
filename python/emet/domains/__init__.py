"""Domain adapters for emet.

Each module constructs a matrix and partition from a specific physical
or mathematical system, ready for emet.decide_dense_matrix().

  torus            Pinched torus, centrifugal barrier, Feller threshold
  yang_mills       SU(2) Kogut-Susskind Hamiltonian (torus instance)
  kahan            Kahan precision envelope for chi certification
  surgery          Post-fracture reconstruction via minimal Markov generator
  mexican_hat      Mexican hat theorem: licensed reductions force potential structure
  torus_4d         4D Yang-Mills as dimension-dependent pinched torus
  yang_mills_sun   SU(N) generalization (SU(2), SU(3)) of Kogut-Susskind
  lattice          Transfer matrix, thermodynamic limit, multi-plaquette scaling
  kuramoto         Kramers-Moyal generator, Hermite mode partition
  graph_laplacian  Two-clique Laplacian, tunable cross-coupling
  transformer      Key cache Gram matrix, eviction mask partition
  quantum_channel  Choi matrix, signal/error partition, Shor-Preskill key rate
  schumann         Earth-ionosphere cavity, thin shell confinement, Schumann spectrum
  ramanujan        Ramanujan hierarchy, spectral gap, chi bound, attention mask
  modular_cusp     Modular surface, hyperbolic Laplacian, cusp partition
  lps_ramanujan    LPS Ramanujan graphs, spectral partition, Fiedler vector
  ihara            Ihara zeta function, zeros from eigenvalues, Re(s) = 1/2
  weil_explicit    Weil explicit formula, cosh envelope, Markov contractivity
  inverse_spectral Gel'fand-Levitan reconstruction, consistency check
"""
