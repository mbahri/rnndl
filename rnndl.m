function [ A, Y, R ] = rnndl( X, k, params )
%NNRDL Implementation of the Robust Non Negative Dictionary Learning
% 
% X: p*n matrix of observations (n observations of dimension p)
% k: size of the dictionary
% Fields in the params structure:
%   * ep: small constant added to stabilize W
%   * beta: beta in the paper
%   * alpha: alpha in the paper
%   * threshold: convergence threshold
%   * MAXITER: maximum number of iterations
%   * y_eps: constant added to Y in the initialization
% 
% Robust Non-Negative Dictionary Learning
% Qihe Pan, Deguang Kong, Chris Ding and Bin Luo
% In Proceedings of the 28th conference of the AAAI - 2014
% 
% k-means initialization uses the PMTK3 toolkit from K. Murphy et. al.
% https://github.com/probml/pmtk3
% From Machine Learning: A Probabilistic Perspective (K. Murphy, 2012)
% 
% Mehdi Bahri
% October, 2016

%% Constants
[~, n] = size(X);
if nargin < 2
    k = n;
end
    
% Epsilon in W
if nargin < 3 || ~isfield(params, 'ep')
    params.ep = eps;
end

% Beta and alpha
if nargin < 3 || ~isfield(params, 'beta')
    params.beta = 0.1;
end

if nargin < 3 || ~isfield(params, 'alpha')
    params.alpha = 1;
end

% E
E = ones(k, n);

% Error threshold for convergence and max number of iterations
if nargin < 3 || ~isfield(params, 'threshold')
    params.threshold = 1e-3;
end

if nargin < 3 || ~isfield(params, 'maxiter')
    params.MAXITER = 1000;
end

%% Initialize the dictionary and the code
[A, Y] = initialize_kmeans(X, k, params);

%% Algorithm
err = inf;
it = 0;
while err > params.threshold && it <= params.MAXITER
    % For convergence
    P = A*Y;
    
    % Update A
    W = 1 ./ sqrt( (X - A*Y).^2 + params.ep .^ 2 );
    A = A .* ( ( (X .* W)*Y' ) ./ ( ((A*Y) .* W)*Y' + 2*params.beta*A ) );
    
    % Update Y: Should it use the same W as A? Comment for comparison
%     W = 1 ./ sqrt( (X - A*Y).^2 + params.ep .^ 2 );
    Y = Y .* ( ( A'*(X .* W) ) ./ (A'*((A*Y) .* W) + params.alpha*E) );
    
    % Compute the relative l2 difference between two consecutive
    % reconstructions
    R = A*Y;
    err = norm(R - P, 'fro') / norm(P, 'fro');
    
    % Display some information
    it = it + 1;
    fprintf('[%d] Delta A*Y = %f\n', it, err);
end

end

function [A, Y, params] = initialize_kmeans(X, k, params)
    % Requires pmtk3 to be installed: https://github.com/probml/pmtk3
    
    if nargin < 3 || ~isfield(params, 'y_eps')
        % In the paper, they add 0.3
        params.y_eps = 0.3;
    end

    nn = size(X, 2);
    [mu, assign, ~] = kmeansFit(X, k);
    A = X * mu;
    Y = zeros(k, nn);
    for i=1:nn
        Y(assign(i), i) = 1;
    end

    Y = Y + params.y_eps;
    
end

