# A0. Aims

Science: differentiable laser-wakefield simulation in Julia, used to optimize a handful of laser and plasma parameters for electron energy, built up a model ladder: 1D wake ODE with a test electron, r-z quasi-static fluid plus envelope, 1D3V PIC, 2D Cartesian PIC with a moving window, then envelope 3D and full 3D on many GPUs. The objective and the gradient-versus-Bayesian question are deliberately still open.

Software: a generic particle–mesh engine with pluggable physics modules (A9), and long term a showcase and stress test for Julia and Reactant compiler technology.

Process: the project is also an experiment in software-factory approaches, with a significant part of the implementation done by autonomous agent teams and loops. The owner sets the aims and the decisions in this directory and monitors the factory; the factory derives everything below them (requirements, interfaces, work packages, tests) and keeps it consistent when an aim or a decision changes. The owner steers by editing these files and vetoes derived work from the daily digest, rather than approving it item by item.
