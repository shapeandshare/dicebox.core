#python setup.py bdist_wheel
#pip install dist/*.whl
python -m unittest discover -s test -p '*Test.py'
