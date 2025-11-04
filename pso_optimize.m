function [globalBestPosition, globalBestValue, history] = pso_optimize(objectiveFcn, dimension, bounds, options)
%PSO_OPTIMIZE Particle Swarm Optimization for bound-constrained problems.
%   [GBEST_POS, GBEST_VAL, HISTORY] = PSO_OPTIMIZE(OBJ, DIM, BOUNDS, OPTS)
%   minimizes the objective function handle OBJ using PSO. DIM is the
%   dimensionality of the search space. BOUNDS is a DIM-by-2 matrix
%   specifying lower and upper bounds for each dimension. OPTS is an
%   optional struct with the following fields:
%       swarmSize            - Number of particles in the swarm (default: 30)
%       maxIterations        - Maximum number of iterations (default: 200)
%       inertiaWeight        - Initial inertia weight (default: 0.9)
%       inertiaDamping       - Multiplicative factor applied to the inertia
%                              weight each iteration (default: 0.99)
%       cognitiveCoeff       - Cognitive acceleration coefficient c1 (default: 2.0)
%       socialCoeff          - Social acceleration coefficient c2 (default: 2.0)
%       velocityClamp        - 1-by-DIM vector with absolute velocity limits.
%                              By default the velocity limit equals half the
%                              decision variable range for each dimension.
%       tolerance            - Stop if the best value improves less than this
%                              threshold between iterations (default: 1e-8)
%       vectorizedObjective  - Set to true when OBJ accepts an N-by-DIM matrix
%                              and returns an N-by-1 vector with one value per
%                              row (default: false)
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
        'tolerance', 1e-8, ...
        'vectorizedObjective', false ...
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
    vectorizedObjective = options.vectorizedObjective;

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

    rangeBounds = upperBounds - lowerBounds;

    % Initialize swarm positions and velocities
    rng('shuffle');
    positions = bsxfun(@plus, lowerBounds, rand(swarmSize, dimension) .* rangeBounds);
    velocities = zeros(swarmSize, dimension);

    personalBestPositions = positions;
    personalBestValues = evaluateSwarm(objectiveFcn, positions, vectorizedObjective);

    [globalBestValue, bestIdx] = min(personalBestValues);
    globalBestPosition = personalBestPositions(bestIdx, :);

    history.bestValues = nan(maxIterations, 1);
    history.inertia = nan(maxIterations, 1);

    lastBestValue = globalBestValue;

    for iteration = 1:maxIterations
        r1 = rand(swarmSize, dimension);
        r2 = rand(swarmSize, dimension);

        velocities = inertiaWeight .* velocities ...
            + c1 .* r1 .* (personalBestPositions - positions) ...
            + c2 .* r2 .* bsxfun(@minus, globalBestPosition, positions);

        % Apply velocity clamping (vectorized)
        velocities = bsxfun(@max, bsxfun(@min, velocities, vClamp), -vClamp);

        positions = positions + velocities;

        % Enforce bounds (vectorized)
        positions = bsxfun(@max, bsxfun(@min, positions, upperBounds), lowerBounds);

        % Evaluate swarm
        currentValues = evaluateSwarm(objectiveFcn, positions, vectorizedObjective);

        improvedMask = currentValues < personalBestValues;
        personalBestValues(improvedMask) = currentValues(improvedMask);
        personalBestPositions(improvedMask, :) = positions(improvedMask, :);

        [iterationBestValue, iterationBestIdx] = min(currentValues);
        if iterationBestValue < globalBestValue
            globalBestValue = iterationBestValue;
            globalBestPosition = positions(iterationBestIdx, :);
        end

        history.bestValues(iteration) = globalBestValue;
        history.inertia(iteration) = inertiaWeight;

        improvement = abs(lastBestValue - globalBestValue);
        lastBestValue = globalBestValue;

        if improvement < tolerance
            break;
        end

        % Update inertia weight
        inertiaWeight = inertiaWeight * inertiaDamping;
    end

    history.bestValues = history.bestValues(1:iteration);
    history.inertia = history.inertia(1:iteration);
end

function values = evaluateSwarm(objectiveFcn, swarm, vectorizedObjective)
    numParticles = size(swarm, 1);
    if vectorizedObjective
        values = objectiveFcn(swarm);
        if isrow(values)
            values = values.';
        end
    else
        values = zeros(numParticles, 1);
        for idx = 1:numParticles
            values(idx) = objectiveFcn(swarm(idx, :));
        end
    end

    if numel(values) ~= numParticles
        error(['Vectorized objective must return one scalar per particle. ', ...
               'Received %d outputs for %d particles.'], numel(values), numParticles);
    end

    values = values(:);
end
