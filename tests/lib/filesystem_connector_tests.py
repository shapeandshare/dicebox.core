import unittest
import filesystem_connector as FileSystemConnectorClass

dataset_location = '/Users/joshburt/Workbench/dicebox.io'

class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    person = FileSystemConnectorClass.init(dataset_location)  # instantiate the Person Class


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
