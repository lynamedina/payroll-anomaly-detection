import sys
print(f"Python executable: {sys.executable}")
try:
    import pandas
    print(f"Pandas location: {pandas.__file__}")
except ImportError:
    print("Pandas is not installed in this environment.")
