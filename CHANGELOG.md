Change Log
==========

03.07.2019
----------
* Updated requirments.
* Added venv to the gitignore list
* Created change log (this!)
* Removed requirements details from readme.
* Updated Copyright dates

03.09.2019
----------
* Fixed docker environment variable over-ride bug which ignored boolean values.

03.10.2019
----------
* Explicitly cast filename to string. Keras model weights save/load issue:  https://github.com/keras-team/keras/issues/11269
* Fixed a bug that could cause the training processor to leave messages on the queues.

03.16.2019
----------
* Added config setting for dicebox compliant datasets.
* Switched over internal logic to use new feature flipper instead of hardcoded list.
* Removed upused variable
* Updated logging
* Corrected configuration override bug

3.17.2019
---------
* Updated logging levels and exception messages.

03.21.2019
----------
* Allow the File System Connector to be optional when creating a Network.

03.22.2019
----------
* Updated classify method to use the dicebox compliance flag in lieu of a magic string.
* Make content indexing optional when creating FSC and Network objects. Scenario is leveraged by classification service currently.
* Cleaning up from exception handling.
