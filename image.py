import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


# кольорове
image = imread('C:\\Users\\Professional\\Downloads\\photo_5393407080783798202_y.jpg')

print(f"Розмір зображення: пікселів по висоті {image.shape[0]}, пікселів по ширині {image.shape[1]}")
print(f"Кількість каналів кольорів: {image.shape[2]}")
plt.figure(figsize=[7,7])
plt.imshow(image)
plt.axis('off')
plt.title('Початкове кольорове зображення')
plt.show()

# чорно-біле

image_sum = image.sum(axis=2)
print(image_sum.shape)

image_bw = image_sum / image_sum.max()
print(image_bw.max())

plt.figure(figsize=[7, 7])
plt.imshow(image_bw, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Чорно-біле зображення')
plt.show()

# графік

pca = PCA()
pca.fit(image_bw)

var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100

k = np.argmax(var_cumu > 95) + 1
print("Number of components explaining 95% variance: " + str(k))

plt.figure(figsize=[10 ,5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.xlabel('Principal Components')
plt.axvline(x=k, color="k", linestyle="--", label=f'{k} components')
plt.axhline(y=95, color="r", linestyle="--", label='95% variance')
plt.plot(var_cumu, label='Cumulative variance')
plt.legend()
plt.show()

# реконструкція

ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

plt.figure(figsize=[7, 7])
plt.imshow(image_recon, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Реконструйоване чорно-біле зображення (95% дисперсії)')
plt.show()


print(f"Max value in reconstructed image: {image_recon.max()}")
print(f"Min value in reconstructed image: {image_recon.min()}")

# реконструкція з різними PCA
def plot_at_k(k):
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))
    plt.imshow(image_recon, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(f'Reconstructed Image with {k} components')
    return image_recon

ks = [10, 25, 50, 100, 150, 250]

plt.figure(figsize=[12, 7])

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plot_at_k(ks[i])
    plt.title("Components: " + str(ks[i]))

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

