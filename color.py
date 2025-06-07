import numpy as np
import cv2
from tkinter import Tk, filedialog, messagebox, simpledialog
import os

# Step 1: Select image file
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select Black and White Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)
if not image_path:
    print("No file selected.")
    exit()

# Step 2: Load model files
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path)
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

# Step 3: Read and process the image
bw_image = cv2.imread(image_path)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]
L_resized = cv2.resize(L, (224, 224))
L_resized -= 50

net.setInput(cv2.dnn.blobFromImage(L_resized))
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_dec_us = cv2.resize(ab_dec, (bw_image.shape[1], bw_image.shape[0]))

L = L[:, :, np.newaxis]
color_lab = np.concatenate((L, ab_dec_us), axis=2)
color_bgr = cv2.cvtColor(color_lab, cv2.COLOR_LAB2BGR)
color_bgr = np.clip(color_bgr, 0, 1)
color_bgr = (255 * color_bgr).astype("uint8")

# Step 4: Display result
cv2.imshow("Original", bw_image)
cv2.imshow("Colorized", color_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 5: Ask user if they want to save
save = messagebox.askyesno("Save Image", "Do you want to save the colorized image?")

if save:
    # Ask for directory to save
    save_dir = filedialog.askdirectory(title="Select Folder to Save Image")
    if save_dir:
        # Build a filename
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, original_filename + "_colorized.png")
        cv2.imwrite(save_path, color_bgr)
        messagebox.showinfo("Saved", f"Image saved to:\n{save_path}")
    else:
        messagebox.showwarning("Canceled", "Save location not selected. Image not saved.")
else:
    print("User chose not to save the image.")
