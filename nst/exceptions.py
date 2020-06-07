import os
class DataLoaderException(Exception):pass

class NoSolutionException(DataLoaderException):pass
class NoSettingsException(DataLoaderException):pass
class InvalidSettingsException(DataLoaderException):pass

def GetAndAssert(dictionary,key):
    value = dictionary.get(key)
    assert value is not None
    return value
def GetAndAssertPath(dictionary,key):
    value = GetAndAssert(dictionary,key)
    assert os.path.exists(value)
    return value
def GetAndAssertInt(dictionary, key):
    value = GetAndAssert(dictionary,key)
    assert int(str(value))==value
    return value