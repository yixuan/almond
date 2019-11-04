#include <Rcpp.h>

using Rcpp::NumericVector;

// [[Rcpp::export]]
NumericVector copula_z(NumericVector u, NumericVector v)
{
    const int n = u.length();
    NumericVector z(n);
    const double* pu = u.begin();
    const double* pv = v.begin();
    double* pz = z.begin();

    for(int i = 0; i < n; i++)
    {
        pz[i] = 0.0;
        for(int j = 0; j < n; j++)
        {
            pz[i] += (pu[j] <= pu[i] && pv[j] <= pv[i]);
        }
        pz[i] /= n;
    }

    return z;
}

