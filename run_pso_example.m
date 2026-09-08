%RUN_PSO_EXAMPLE Demonstration of the PSO optimizer on the Rastrigin function.
%
%   To run the example execute this script in MATLAB or GNU Octave. The
%   script optimizes the 5-dimensional Rastrigin function and plots the
%   convergence curve.

% Objective function (Rastrigin). Supports vectorized evaluation where rows
% correspond to particles.
rastrigin = @(x) 10 * size(x, 2) + sum(x.^2 - 10 * cos(2 * pi * x), 2);

% Problem definition
numVariables = 5;
variableBounds = repmat([-5.12, 5.12], numVariables, 1);

% Algorithm options
options = struct( ...
    'swarmSize', 40, ...
    'maxIterations', 150, ...
    'inertiaWeight', 0.8, ...
    'inertiaDamping', 0.98, ...
    'cognitiveCoeff', 1.8, ...
    'socialCoeff', 2.0, ...
    'tolerance', 1e-12, ...
    'vectorizedObjective', true ...
);

% Run optimization
[bestPosition, bestValue, history] = pso_optimize(rastrigin, numVariables, variableBounds, options);

% Display results
fprintf('Best value found: %.6f\n', bestValue);
fprintf('Best position: %s\n', mat2str(bestPosition, 6));

% Plot convergence history if available
if isfield(history, 'bestValues') && numel(history.bestValues) > 1
    figure;
    semilogy(history.bestValues, 'LineWidth', 1.5);
    grid on;
    xlabel('Iteration');
    ylabel('Best objective value (log scale)');
    title('PSO convergence on the Rastrigin function');
end
