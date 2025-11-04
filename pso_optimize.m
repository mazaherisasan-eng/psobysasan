function [globalBestPosition, globalBestValue, history] = pso_optimize(objectiveFcn, dimension, bounds, options)
%PSO_OPTIMIZE Particle Swarm Optimization for bound-constrained problems.
%   [GBEST_POS, GBEST_VAL, HISTORY] = PSO_OPTIMIZE(OBJ, DIM, BOUNDS, OPTS)
%   minimizes the objective function handle OBJ using PSO. DIM is the
%   dimensionality of the search space. BOUNDS is a DIM-by-2 matrix
%   specifying lower and upper bounds for each dimension. OPTS is an
%   optional struct with the following fields:
%       swarmSize       - Number of particles in the swarm (default: 30)
%       maxIterations   - Maximum number of iterations (default: 200)
%       inertiaWeight   - Initial inertia weight (default: 0.9)
%       inertiaDamping  - Multiplicative factor applied to the inertia
%                         weight each iteration (default: 0.99)
%       cognitiveCoeff  - Cognitive acceleration coefficient c1 (default: 2.0)
%       socialCoeff     - Social acceleration coefficient c2 (default: 2.0)
%       velocityClamp   - 1-by-DIM vector with absolute velocity limits.
%                         By default the velocity limit equals half the
%                         decision variable range for each dimension.
%       tolerance       - Stop if the best value improves less than this
%                         threshold between iterations (default: 1e-8)
%
%   HISTORY is a struct with fields:
%       bestValues      - Best objective value found at each iteration
%       inertia         - Inertia weight used at each iteration
%
%   Example:
%       sphere = @(x) sum(x.^2);
%       dim = 5;
%       bounds = repmat([-5.12, 5.12], dim, 1);
%       [pos, val, hist] = pso_optimize(sphere, dim, bounds, struct('maxIterations', 100));
%
%   This implementation supports both MATLAB and GNU Octave.

    if nargin < 4
        options = struct();
    end

    % Validate inputs
    if ~isa(objectiveFcn, 'function_handle')
        error('objectiveFcn must be a function handle.');
    end
    if ~isscalar(dimension) || dimension <= 0
        error('dimension must be a positive scalar.');
    end
    if size(bounds, 1) ~= dimension || size(bounds, 2) ~= 2
        error('bounds must be a DIM-by-2 matrix.');
    end

    lowerBounds = bounds(:, 1)';
    upperBounds = bounds(:, 2)';
    if any(lowerBounds >= upperBounds)
        error('Each lower bound must be strictly less than the corresponding upper bound.');
    end

    % Default options
    defaults = struct(
        'swarmSize', 30, ...
        'maxIterations', 200, ...
        'inertiaWeight', 0.9, ...
        'inertiaDamping', 0.99, ...
        'cognitiveCoeff', 2.0, ...
        'socialCoeff', 2.0, ...
        'velocityClamp', [], ...
        'tolerance', 1e-8 ...
    );

    defaultsFields = fieldnames(defaults);
    for k = 1:numel(defaultsFields)
        field = defaultsFields{k};
        if ~isfield(options, field) || isempty(options.(field))
            options.(field) = defaults.(field);
        end
    end

    swarmSize = options.swarmSize;
    maxIterations = options.maxIterations;
    inertiaWeight = options.inertiaWeight;
    inertiaDamping = options.inertiaDamping;
    c1 = options.cognitiveCoeff;
    c2 = options.socialCoeff;
    vClamp = options.velocityClamp;
    tolerance = options.tolerance;

    if isempty(vClamp)
        vClamp = 0.5 * (upperBounds - lowerBounds);
    else
        if numel(vClamp) == 1
            vClamp = repmat(abs(vClamp), 1, dimension);
        elseif numel(vClamp) ~= dimension
            error('velocityClamp must be scalar or a vector with DIM elements.');
        end
        vClamp = abs(vClamp(:))';
    end

    % Initialize swarm positions and velocities
    rng('shuffle');
    positions = repmat(lowerBounds, swarmSize, 1) + rand(swarmSize, dimension) .* (repmat(upperBounds - lowerBounds, swarmSize, 1));
    velocities = zeros(swarmSize, dimension);

    personalBestPositions = positions;
    personalBestValues = arrayfun(@(idx) objectiveFcn(positions(idx, :)), 1:swarmSize);

    [globalBestValue, bestIdx] = min(personalBestValues);
    globalBestPosition = personalBestPositions(bestIdx, :);

    history.bestValues = nan(maxIterations, 1);
    history.inertia = nan(maxIterations, 1);

    iteration = 0;
    lastBestValue = globalBestValue;

    while iteration < maxIterations
        iteration = iteration + 1;

        r1 = rand(swarmSize, dimension);
        r2 = rand(swarmSize, dimension);

        velocities = inertiaWeight .* velocities ...
            + c1 .* r1 .* (personalBestPositions - positions) ...
            + c2 .* r2 .* (repmat(globalBestPosition, swarmSize, 1) - positions);

        % Apply velocity clamping
        for d = 1:dimension
            velocities(:, d) = max(min(velocities(:, d), vClamp(d)), -vClamp(d));
        end

        positions = positions + velocities;

        % Enforce bounds
        for d = 1:dimension
            positions(:, d) = max(min(positions(:, d), upperBounds(d)), lowerBounds(d));
        end

        % Evaluate swarm
        for i = 1:swarmSize
            currentValue = objectiveFcn(positions(i, :));
            if currentValue < personalBestValues(i)
                personalBestValues(i) = currentValue;
                personalBestPositions(i, :) = positions(i, :);
            end
            if currentValue < globalBestValue
                globalBestValue = currentValue;
                globalBestPosition = positions(i, :);
            end
        end

        history.bestValues(iteration) = globalBestValue;
        history.inertia(iteration) = inertiaWeight;

        % Update inertia weight
        inertiaWeight = inertiaWeight * inertiaDamping;

        if abs(lastBestValue - globalBestValue) < tolerance
            history.bestValues = history.bestValues(1:iteration);
            history.inertia = history.inertia(1:iteration);
            break;
        end
        lastBestValue = globalBestValue;
    end

    % Trim history if convergence reached early
    if iteration == maxIterations
        history.bestValues = history.bestValues(1:maxIterations);
        history.inertia = history.inertia(1:maxIterations);
    end
end
