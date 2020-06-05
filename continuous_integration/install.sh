if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
  wget https://repo.continuum.io/miniconda/Miniconda${TRAVIS_PYTHON_VERSION:0:1}-latest-Linux-x86_64.sh -O miniconda.sh;
elif [ "${TRAVIS_OS_NAME}" == "osx" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda${TRAVIS_PYTHON_VERSION:0:1}-latest-MacOSX-x86_64.sh -O miniconda.sh;
fi

bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
# Create virtual envs with fixed dependencies
conda create -q -n test-env python=${TRAVIS_PYTHON_VERSION}
conda activate test-env

#Build and install the package inside the conda environnement
if [ "${INSTALL}" == "pip" ]; then
  if [ "${ENV}" == 'default-user' ]; then
    pip install pytest
    pip install .
  elif [ "${ENV}" == 'lining' ]; then
    pip install autopep8 flake8
  else
    pip install pytest
    pip install .[${ENV}]
  fi
elif [ "${INSTALL}" == "conda" ]; then
 conda build conda-recipe
 conda install slam --use-local
fi
