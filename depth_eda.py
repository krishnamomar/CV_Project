import numpy as np
import matplotlib.pyplot as plt


def depth_eda(image_left, depth_gt):
    # Load the image and depth data for a specific scene

    # Display the left image and the ground-truth depth map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_left, cmap='gray')
    axes[1].imshow(depth_gt, cmap='jet')
    axes[0].set_title("Left Image")
    axes[1].set_title("Ground-Truth Depth")
    plt.show()

    # Compute some basic statistics on the depth data
    print("Depth Statistics:")
    print(f"   Min: {depth_gt.min()}")
    print(f"   Max: {depth_gt.max()}")
    print(f"   Mean: {depth_gt.mean()}")
    print(f"   Median: {np.median(depth_gt)}")
    print(f"   Std: {depth_gt.std()}")

    # Plot the distribution of depth values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(depth_gt.flatten(), bins=100, density=True, alpha=0.5)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Depth Distribution")
    plt.show()

    # Compute the gradient magnitude of the depth map and display it
    depth_grad = np.gradient(depth_gt)
    depth_grad_mag = np.sqrt(depth_grad[0]**2 + depth_grad[1]**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(depth_grad_mag, cmap='gray')
    ax.set_title("Depth Gradient Magnitude")
    plt.show()
