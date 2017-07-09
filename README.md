# dicebox
               Let's shake things up!

Mission Statement / Why
-----------------------
I have a simple dream.  A dream of one day playing Dugeons and Dragons live online and have REAL dice involved.  There are lots of great programs with shared dice rollers in them.  I know.  What I want is the real deal.  So I began crafting dicebox.

Dicebox has come through a number of iterations, but finally landed on the implementation and code heritage it has based on what works, is widely available, and accepted for this problem domain.

Thanks everyone!

Overview / Abstract
-------------------
A robust trainable image classification system.

1. Visual Image Classification

    Dicebox is a visual classification system.


2. Evolutionary Neural Network

    Dicebox is capable of being applied to a large variety of classification problems.  Sometimes unique or novel problems need to be solved an a neural network structure unknown.  In this case dicebox provides a means to evolve a network tailed to the particular problem.

3. Service-oriented Architecture

  The trained classification system is access through a REST API.
  
Audience
--------
Those who need automated dice roll recognition, or wish to use dicebox on another data set.

Quick Start
-----------
```
    cd {project_root}
    source ~/tensorflow/bin/active
    pip install -r requirements.txt
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
[Download](https://s3-us-west-2.amazonaws.com/diceboximages/dist/dicebox_60x50.070817.tar.gz)
 | Dicebox Dataset 60x50 | approx 265k  gray scale png images of 1d4 and 1d6 dices.
     
Configuration
=============
`dicebox.config` contains the configurable parameters for dicebox.

The below section controls the parameters for the network input.
```
[DATASET]
    name = dicebox
    categories = 11
    image_width = 480
    image_height = 270
```

Images are expected to be in the below directory structure.
```
(dataset_base_directory)\
    (name)_(width)x(height)\
        data\
            (category_1)\
            (category_2)\
            [..]
```

When evolution is underway in the primordial pool the below parameters control the options for the networks that can be appear within individuals.
```
[TAXONOMY]
    neurons: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    layers: [1, 2, 3, 5, 8, 13, 21]
    activation: ["relu", "elu", "tanh", "sigmoid"]
    optimizer: ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]
```

Individual networks can be defined and trained.  This controls the network that is trained and used by the service.
```
[LONESTAR]
    neurons = 987
    layers = 2
    activation = sigmoid
    optimizer = adamax
```

The number of epochs, individuals, and generations used during evolution are defined a below.
```
[EVOLUTION]
    epochs = 10000
    generations = 100
    population = 50
```

When training the lonestar network, the below options control the training regiment.
```
[TRAINING]
    batch_size = 500
    train_batch_size = 5000
    test_batch_size = 500
```

Defines the directoried used for the various file system operations.
```
[DIRECTORY]
    dataset_base_directory = datasets
    logs_dir = ./logs
    weights_dir = ./weights
    tmp_dir = ./tmp
```

Controls the service configuration.
```
[SERVER]
    api_access_key = 6e249b5f-b483-4e0d-b50b-81d95e3d9a59
    api_version = 0.2.1
    listening_host = 0.0.0.0
    flask_debug = False
    model_weights_filename = weights.best.hdf5
```

Noise, luck, whatever it is.  Here's its control.
```
[GLOBAL]
    noise = 0.1
```

Service configuration for the dicebox client.
```
[CLIENT]
    classification_server = localhost
    classification_port = 5000
    uri = http://
```

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
Provides an end-point that performs classifications via REST API calls.

Start the service:
```
    python ./dicebox_service.py
```

Dicebox API
===========

Default URL for API: `http(s)://{hostname}:5000/`

Anonymous End-Points
-------------------

**Get Service API Version**

For verification of service API version.

```
    [GET] /api/version
```

Result:
`
{
    "version": "String"
}
`

**Get Service Health**
 
For use in load balanced environments.

```
    [GET] /health/plain
```
Result:
`true` or `false`


Authentication Required End-Points
-----------------------

**Request Header**

The below end-points require several host headers to be present on the request.

```
    'Content-type': 'application/json',
    'API-ACCESS-KEY': 'String',
    'API-VERSION': 'String'
```

* API-ACCESS-KEY: 'Guid used for authorization'
* API-VERSION: 'Version of the API to use'

**Classification**

Used to classify the image data.  Return the label index for the classification.


```
    [POST] /classify
```
Post Body: `{ "data": "Base64 encoded PNG image" }`

Result: `{ "classification": integer }`


Example
```
    {
        "classification": 7
    }

```
**Get Categories**

 Used to turn classification results into human-readable labels.

```
    [GET] /categories
```
Result: A list of label to index mappings.

Example

```
    {
        "category_map": {
            "0": "1d4_1",
            "1": "1d4_2",
            "2": "1d4_3",
            "3": "1d4_4",
            "4": "1d6_1",
            "5": "1d6_2",
            "6": "1d6_3",
            "7": "1d6_4",
            "8": "1d6_5",
            "9": "1d6_6",
            "10": "unknown"
        }
    }
```

Client Consumption
==================
Sample client application.  Useful for supervised training.
```
    python ./client/dicebox_client.py
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