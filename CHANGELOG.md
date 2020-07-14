Change Log
==========

3.1.0
-----
- Updated exports for external usage.
- Addressed issue in evolve and serialized networks

3.0.0
-----
- Updated interface, and code layout.

2.1.2 (07/04/2020)
-----
- Addressed a cross-platform seperator bug.

2.1.1 (07/04/2020)
-----
- Address old import.

2.1.0 (07/03/2020)
-----
- Refactored from Python 2.7 to 3.7.
- Added support for distribution as a wheel package.

2.0.1
-----
- Bug fix, removed unimplemented exceptions left after refactoring.

2.0.0
-----
- Neural networks can now have individual layer lengths, types, activation functions.

04.06.2019
----------
* Reactored variables names
* Removed unused imports
* Removed unused objects
* Added additional test coverage
* Moved assets, and tools to the SDK.
* Refactored namespace

03.31.2019
----------
* Fixed bug which prevent luck from working correctly.
* Disabled code which cached data to disk for the training processor until the code to pull from cache is also completed.

03.30.2019
----------
* Refactored configuration into a class to allow config file over-rides
* Initial round of linting completed
* Unit tests for file system connector class added.

03.23.2019
----------
* Fix leaking file descriptors when polling for queue readiness within the trainingprocessor / sensory interface code.

03.22.2019
----------
* Updated classify method to use the dicebox compliance flag in lieu of a magic string.

03.21.2019
----------
* Allow the File System Connector to be optional when creating a Network.

3.17.2019
---------
* Updated logging levels and exception messages.

03.16.2019
----------
* Added config setting for dicebox compliant datasets.
* Switched over internal logic to use new feature flipper instead of hardcoded list.
* Removed unused variable
* Updated logging
* Corrected configuration override bug

03.10.2019
----------
* Explicitly cast filename to string. Keras model weights save/load issue:  https://github.com/keras-team/keras/issues/11269
* Fixed a bug that could cause the training processor to leave messages on the queues.

03.09.2019
----------
* Fixed docker environment variable over-ride bug which ignored boolean values.

03.07.2019
----------
* Updated requirments.
* Added venv to the gitignore list
* Created change log (this!)
* Removed requirements details from readme.
* Updated Copyright dates
