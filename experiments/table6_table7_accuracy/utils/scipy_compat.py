def patch_geoopt_linesearch_import() -> None:
    import scipy.optimize.linesearch as public_linesearch
    from scipy.optimize import _linesearch

    if not hasattr(public_linesearch, "scalar_search_wolfe2"):
        public_linesearch.scalar_search_wolfe2 = _linesearch.scalar_search_wolfe2
    if not hasattr(public_linesearch, "scalar_search_armijo"):
        public_linesearch.scalar_search_armijo = _linesearch.scalar_search_armijo
