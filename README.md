# dicebox
               ..let's shake things up..

Mission Statement / Why
-----------------------
I have a simple dream.  A dream of one day playing Dugeons and Dragons on YouTube and have REAL dice involved.  There are lots of great programs with shared dice rollers in them.  I know.  What I want is the real deal.  So I began crafting dicebox.

Dicebox has come through a number of iterations, but finally landed on the implementation and code heritage it has based on what works, is widely available, and accepted for this problem domain.

Thanks everyone!

Overview / Abstract
-------------------
Trainable image classification system capable of predicting through a REST API.

1. Visual Image Classification
2. Evolutionary Neural Network
3. Service Oriented Architecture


Audience
--------
Those who need automated dice roll recognition, or wish to use dicebox on another data set.

Quick Start
-----------
```
screen
cd {project_root}
pip install -r requirements.txt
source ~/tensorflow/bin/active
```

Requirements
------------

**Python Module Requirements**

The requirements can be automatically installed using the below command:
```
pip install -r requirements.txt
```
These are the individual modules required by this project:

```
requests==2.13.0
numpy==1.11.0
Flask==0.12.1
tqdm==4.14.0
tensorflow==1.1.0
Keras==2.0.4
Pillow==4.2.0
```

Data Sets
---------
Dataset currently exceeds 100k images and is growing constaintly.

Dicebox Dataset | 60x50 gray scale png files.

Configuration
=============
`lib\dicebox_config.py` contains the hyper-parameters for dicebox.


The Primordial Pool
===================
Based on the defined hyper-parameters will create successive generations of neural networks.  Evolutional optimization is employs to select for successive generations.
```
python ./primordialpool.py
```

Lonestar Training
=================
Allows for the saving of specific networks pulled from the pool.  The trained weights will be saved for use later.
```
python ./lonestar_train.py
```

Classification via REST API
===========================
Creates the lonestar model, loads the weights, and then performs classifications via REST API calls.

Start the service:
```
python ./dicebox_service.py
```

Dicebox API
===========

`http://{host}:5000/`


Anonymous End-Points
-------------------
`/api/version`

`/health/plain`

Authentication Required
-----------------------
`/prediction`


Client Consumption
==================
For an interactive experience:
```
python ./client/dicebox_client.py
```

For rapidly testing a large set of images:
```
python ./client/dicebox_test_client.py
```

Contributing
------------
1. Fork the repository on Github
2. Create a named feature branch (like `add_component_x`)
3. Write your change
4. Write tests for your change (if applicable)
5. Run the tests, ensuring they all pass
6. Submit a Pull Request using Github

License and Authors
-------------------
MIT License

Copyright (c) 2017 Joshua C. Burt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.