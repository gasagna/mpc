from distutils.core import setup, Extension


setup(  name        = "pympc",
        author      =  "Davide Lasagna",
        author_email = "davide.lasagna@polito.it",
        license     = "GPL",
        url         = 'None',
        version     = "0.0.1",
        description = "A module for simulation of discrete linear time invariant systems controlled with Model Predictive Controllers",
        packages = ['mpc'], 
        )
