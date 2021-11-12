# https://developer.apple.com/metal/tensorflow-plugin/
# https://betterprogramming.pub/installing-tensorflow-on-apple-m1-with-new-metal-plugin-6d3cb9cb00ca
conda create --name tensorflow_m1 python==3.9 -y
conda activate tensorflow_m1

# remove an environment
# conda env remove -n tensorflow_m1

# build dependencies
conda install wheel setuptools build -y

# test dependencies
conda install pylint black coverage python-dotenv safety -y
pip install python-mnist

# primary dependencies
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos
pip install tensorflow-metal
conda install pillow configparser pika requests tqdm -y
