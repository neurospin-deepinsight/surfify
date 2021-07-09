# -*- coding: utf-8 -*-
"""
UNet segmentation
=================

Credit: A Grigis

A simple example on how to use the SphericalUNet architecture on the
classification dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from surfify.utils import icosahedron
from surfify.plotting import plot_trisurf
from surfify.models import SphericalUNet
from surfify.datasets import make_classification, ClassificationDataset

#############################################################################
# Inspect dataset
# ---------------
#
# First we load the classification dataset (with 3 classes) and inspect the
# genrated labels.

ico_order = 3
n_classes = 3
n_epochs = 20
ico_vertices, ico_triangles = icosahedron(order=ico_order)
n_vertices = len(ico_vertices)
X, y = make_classification(ico_order, n_samples=40, n_classes=n_classes,
                           scale=1, seed=42)
print("Surface:", ico_vertices.shape, ico_triangles.shape)
print("Data:", X.shape, y.shape)
plot_trisurf(ico_vertices, ico_triangles, y, is_label=True)

#############################################################################
# Train the model
# ---------------
#
# We now train the SphericalUNet model using a CrossEntropy loss and a SGD
# optimizer. As it is obvious to segment the input classification dataset
# an accuracy of 100% is expected.

dataset = ClassificationDataset(
    ico_order, n_samples=40, n_classes=n_classes, scale=1, seed=42)
loader = DataLoader(dataset, batch_size=5, shuffle=True)
model = SphericalUNet(
    in_order=ico_order, in_channels=n_classes, out_channels=n_classes,
    depth=2, start_filts=8, conv_mode="1ring", up_mode="transpose")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.99, weight_decay=1e-4)
size = len(loader.dataset)
n_batches = len(loader)
for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch * len(X)
        if epoch % 5 == 0:
            print("loss {0}: {1:>7f}  [{2:>5d}/{3:>5d}]".format(
                epoch, loss, current, size))
model.eval()
test_loss, correct = 0, 0
y_preds = []
with torch.no_grad():
    for X, y in loader:
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        logit = torch.nn.functional.softmax(pred, dim=1)
        y_pred = pred.argmax(dim=1)
        correct += (y_pred == y).type(torch.float).sum().item()
        y_preds.append(y_pred.numpy())
test_loss /= n_batches
correct /= (size * n_vertices)
y_preds = np.concatenate(y_preds, axis=0)
print("Test Error: \n Accuracy: {0:>0.1f}%, Avg loss: {1:>8f}".format(
    100 * correct, test_loss))


#############################################################################
# Inspect the predicted labels
# ----------------------------
#
# Finally the predicted labels of the first sample are displayed. As expected
# they corresspond exactly to the ground truth.

plot_trisurf(ico_vertices, ico_triangles, y_pred[0], is_label=True)
plt.show()
