from matplotlib import pyplot as plt
from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")
print(ds.num_rows)

flower = ds["train"][15]["image"]
plt.imshow(flower)
plt.axis("off")
plt.show()
