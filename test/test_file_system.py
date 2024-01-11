import unittest
from OCC.Core.TopoDS import TopoDS_Compound

from src.model_3d.file_system import FileReader


class FileReaderTests(unittest.TestCase):
    def test_read_stp_file(self):
        file_name = "AirCompressor.stp"

        shape = FileReader.read_stp_file(file_name)

        self.assertIsInstance(shape, TopoDS_Compound)
        return


if __name__ == '__main__':
    unittest.main()