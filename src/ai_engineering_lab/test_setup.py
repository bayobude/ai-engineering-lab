import numpy as np
import pandas as pd

numbers = np.array([10, 20, 30, 40, 50])

data = pd.DataFrame({
    "Number": numbers,
    "Squared": numbers ** 2
})

print(data)
print("\nAI Engineering setup is working from VS Code!")