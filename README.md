# psobysasan

This repository provides a simple MATLAB / GNU Octave implementation of the
Particle Swarm Optimization (PSO) metaheuristic along with a demo script.

## Files

- `pso_optimize.m` – Generic PSO solver for bound-constrained optimization
  problems. Configure the algorithm through the `options` structure to control
  swarm size, coefficients, inertia damping, convergence tolerance, velocity
  limits, and whether the objective function can be evaluated in a vectorized
  fashion for additional speed.
- `run_pso_example.m` – Example script that minimizes the 5D Rastrigin
  benchmark function using the optimizer and plots the convergence history.

## Usage

1. Open MATLAB or GNU Octave in this repository directory.
2. Run the example script:

   ```matlab
   run_pso_example
   ```

   The script prints the best solution found and displays a convergence plot.

3. To use the optimizer for your own problem, call `pso_optimize` with a
   function handle, the number of decision variables, their bounds, and an
   optional options structure:

   ```matlab
   myObjective = @(x) ... % compute scalar cost
   dimension = 3;
   bounds = [0 1; -5 5; -10 10];
   opts = struct('swarmSize', 50, 'maxIterations', 250);
   [bestPos, bestVal] = pso_optimize(myObjective, dimension, bounds, opts);
   ```

Commonly tuned options include `swarmSize`, `cognitiveCoeff`, `socialCoeff`,
`inertiaWeight`, and `tolerance`. For expensive objective functions that can
operate on many particles at once, set `vectorizedObjective` to `true` and
implement your objective to accept an `N`-by-`DIM` matrix of particles and
return an `N`-by-1 vector of objective values to reduce function call
overheads. Adjust the coefficients and tolerances in the options structure to
match your specific optimization problem.
