rm *.c
rm -rf build 
export cythonCC="gcc"
python3 setup.py build_ext --inplace
python3 setup_gp.py build_ext --inplace
python3 setup_epsilon.py build_ext --inplace
python3 setup_post_epsilon.py build_ext --inplace
python3 setup_blocked_gibbs.py build_ext --inplace
python3 setup_gaussian_emission.py build_ext --inplace


