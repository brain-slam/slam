if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
# Create virtual envs with fixed dependencies
conda create -q -n test-env python=$TRAVIS_PYTHON_VERSION cython numpy matplotlib flake8 autopep8 pytest pytest-cov coveralls
conda install -n test-env -c conda-forge nibabel trimesh
conda  install -n test-env -c mmwoodman gdist
conda activate test-env
python ../setup.py install
