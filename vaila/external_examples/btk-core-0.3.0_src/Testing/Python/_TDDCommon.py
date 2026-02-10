import unittest

import AcquisitionTest
import AnalogTest
import EventCollectionTest
import EventTest
import ForcePlatformTypesTest
import IMUTypesTest
import MetaDataInfoTest
import MetaDataTest
import PointCollectionTest
import PointTest
import WrenchTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(EventTest.EventTest))
    suite.addTest(unittest.makeSuite(AnalogTest.AnalogTest))
    suite.addTest(unittest.makeSuite(PointTest.PointTest))
    suite.addTest(unittest.makeSuite(ForcePlatformTypesTest.ForcePlatformTypesTest))
    suite.addTest(unittest.makeSuite(EventCollectionTest.EventCollectionTest))
    suite.addTest(unittest.makeSuite(PointCollectionTest.PointCollectionTest))
    suite.addTest(unittest.makeSuite(MetaDataInfoTest.MetaDataInfoTest))
    suite.addTest(unittest.makeSuite(MetaDataTest.MetaDataTest))
    suite.addTest(unittest.makeSuite(AcquisitionTest.AcquisitionTest))
    suite.addTest(unittest.makeSuite(WrenchTest.WrenchTest))
    suite.addTest(unittest.makeSuite(IMUTypesTest.IMUTypesTest))
    return suite
