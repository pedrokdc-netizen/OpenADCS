from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "openadcs_sim._xyzgeomag",
            sources=["simulation/native/xyzgeomag.cpp"],
            include_dirs=["simulation/vendor/xyzgeomag"],
            language="c++",
        )
    ]
)
