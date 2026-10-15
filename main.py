import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import joblib
import streamlit
import paho.mqtt.client as mqtt


def check_environment():

    print("=" * 55)
    print("AIoT MOTOR BEARING FAULT DETECTION")
    print("Environment Verification")
    print("=" * 55)

    print("\nPython Version:")
    print(sys.version)

    print("\nInstalled Libraries:")

    libraries = {
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "SciPy": scipy.__version__,
        "Scikit-learn": sklearn.__version__,
        "Matplotlib": matplotlib.__version__,
        "Joblib": joblib.__version__,
        "Streamlit": streamlit.__version__,
    }

    for name, version in libraries.items():
        print(f"{name:20}: {version}")

    print("\nMQTT Library: Installed")

    print("\n" + "=" * 55)
    print("PROJECT ENVIRONMENT READY!")
    print("=" * 55)


if __name__ == "__main__":
    check_environment()