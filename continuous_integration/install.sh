if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
  wget https://repo.continuum.io/miniconda/Miniconda${TRAVIS_PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
  #small hack to force the python version to default value tested since travis does not fully support python in macosx
  TRAVIS_PYTHON_VERSION="3.6"
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
conda activate test-env
pip install gdist
python setup.py install
