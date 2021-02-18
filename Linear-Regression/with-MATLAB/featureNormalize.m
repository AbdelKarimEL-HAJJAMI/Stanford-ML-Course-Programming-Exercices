function [X_norm, mu, sigma] = featureNormalize(X)
%   FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
     
mu = mean(X);                   % mean of each feature of X (i.e. of each column)
sigma = std(X);                 % a row containing the standard deviation of every feature of X
X_norm = (X-mu)./sigma;         % feature scaling


end
