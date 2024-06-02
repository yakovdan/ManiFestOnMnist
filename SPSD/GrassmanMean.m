function mMeanP = GrassmanMean(GG, vW)
    N       = length(GG);
    if nargin < 2
        vW = ones(N, 1) / N;
    end
    vW      = vW / sum(vW);
    [D, d]  = size(GG{1});
    M       = grassmannfactory(D, d, 1);
    
    mMeanP  = GG{1};
    maxIter = 200;
    vNorm   = nan(maxIter, 1);
    for ii = 1 : maxIter
        
        mLogMean = 0 * mMeanP;
        for nn = 1 : N
            mLogMean = mLogMean + vW(nn) * M.log(mMeanP, GG{nn});
        end
        mMeanP = M.exp(mMeanP, mLogMean);
        
        vNorm(ii) = norm(mLogMean, 'fro');
        fprintf("Grassmann Norm %.12f %d\n", vNorm(ii), ii);
        if vNorm(ii) < 1e-10
            fprintf("Grassmann Norm %.12f %d\n", vNorm(ii), ii);
            break
        end
    end
    fprintf("Grassmann Norm %.12f\n", vNorm(ii));
%     figure; plot(log(vNorm)); title('Norm of mean - should be zero at the end');
end