#!/usr/bin/python

from setuptools import setup

setup(
    name="tvm",
    version="0.1",
    packages=["tvm", "tvm._ffi", "tvm._ffi._ctypes", "tvm.contrib"],
    package_data={"tvm": ["*.so"]},
    entry_points="",
)
