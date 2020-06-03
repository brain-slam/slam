# style check
if [ "${ENV}" == 'test' ] || [ "${ENV}" == 'full' ]; then
  flake8 slam/
  flake8 examples/
  flake8 tests/
  autopep8 --recursive --aggressive --diff --exit-code slam/
  autopep8 --recursive --aggressive --diff --exit-code examples/
  autopep8 --recursive --aggressive --diff --exit-code tests/
  #- pytest
  pytest -W default --cov=slam/
fi