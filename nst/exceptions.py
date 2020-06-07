import os
class DataLoaderException(Exception):pass

class NoSolutionException(DataLoaderException):pass
class NoSettingsException(DataLoaderException):pass
class InvalidSettingsException(DataLoaderException):pass
class InvalidDataException(DataLoaderException):pass

class NSTException(Exception):pass

class NoSuchLayerException(NSTException):pass
class InvalidHCoefficientException(NSTException):pass

def GetAndAssert(dictionary,key):
    value = dictionary.get(key)
    assert (value is not None,"Key {0} is not exist".format(key))
    return value
def GetAndAssertPath(dictionary,key,additionalPath = ""):
    value = additionalPath+"/"+GetAndAssert(dictionary,key)
    assert (os.path.exists(value),"Path {0} is not exist".format(value))
    return value
def GetAndAssertInt(dictionary, key):
    value = GetAndAssert(dictionary,key)
    assert (isinstance(value,int),"Key {0} is not an int".format(key))
    return value
def GetAndAssertFloat(dictionary, key):
    value = GetAndAssert(dictionary,key)
    assert (isinstance(value,float),"Key {0} is not a float".format(key))
    return value