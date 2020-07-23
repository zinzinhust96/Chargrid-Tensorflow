import matplotlib.pyplot as plt
import numpy as np

def compare_input_augmented_input(index_to_test, trainset, batch_chargrid, batch_seg, batch_mask, batch_coord):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_chargrid_1h, trainset[index_to_test]))))
    ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_chargrid[index_to_test]))
    plt.show()
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_gt_1h, trainset[index_to_test]))))
    ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_seg[index_to_test]))
    plt.show()
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_mask, trainset[index_to_test]))[:, :, 0])
    ax2.imshow(batch_mask[index_to_test][:, :, 0])
    plt.show()
    plt.clf()

    print(batch_coord[index_to_test][(batch_coord[index_to_test] > 1e-6)[:, :, 0], 0]*width)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_coord, trainset[index_to_test]))[:, :, 0])
    ax2.imshow(batch_coord[index_to_test][:, :, 0])
    plt.show()
    plt.clf()

def plot_loss(history_train, history_test, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_train, label="loss")
    plt.plot(history_test, label="val_loss")
    plt.plot(np.argmin(history_test), np.min(history_test), marker="o", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()

def plot_time(history_time, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_time, label="time")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()