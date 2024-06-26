function [mC, mG, mP, UU, TT] = SpsdMean(CC, r, mG)
    display('SPSD MEAN');
    Symm  = @(M) (M + M') / 2;

    N     = length(CC);
    UU{N} = [];
    
    for ii = 1 : N
        Xi          = Symm(CC{ii});
        [UU{ii}, ~] = eigs(Xi, r);
    end

    if nargin < 3
        mG = GrassmanMean(UU);
    end
    
    TT{N} = [];
    for ii = 1 : N
        Xi           = CC{ii};
        Ui           = UU{ii};
        [Oi, ~, OWi] = svd(Ui' * mG);
        GOi          = Ui * Oi * OWi';
        Ti           = GOi' * Xi * GOi;
        TT{ii}       = Symm(Ti);
    end
    
    mP = SpdMean(TT);
    mC = Symm(mG * mP * mG');
end