import unittest

import btk
import numpy


class PointTest(unittest.TestCase):
    def test_Constructor(self):
        test = btk.btkPoint("HEEL_R", 200)
        self.assertEqual(test.GetLabel(), "HEEL_R")
        self.assertEqual(test.GetDescription(), "")
        self.assertEqual(test.GetValues().shape[0], 200)
        self.assertEqual(test.GetValues().shape[1], 3)
        self.assertEqual(test.GetResiduals().shape[0], 200)
        self.assertEqual(test.GetType(), btk.btkPoint.Marker)

    def test_FrameNumber(self):
        test = btk.btkPoint("HEEL_R", 200)
        test.SetFrameNumber(225)
        self.assertEqual(test.GetValues().shape[0], 225)
        self.assertEqual(test.GetResiduals().shape[0], 225)

    def test_SetValues(self):
        p = btk.btkPoint("HEEL_R", 4)
        values = numpy.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        p.SetValues(values)
        values_extracted = p.GetValues()
        for i in range(0, 4):
            self.assertEqual(values_extracted[i, 0], values[i, 0])
            self.assertEqual(values_extracted[i, 1], values[i, 1])
            self.assertEqual(values_extracted[i, 2], values[i, 2])
