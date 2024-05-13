import datetime
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.data import shepp_logan_phantom
from skimage.draw import ellipse as circle, line
from skimage.color import gray2rgb, rgb2gray
from scipy.fftpack import fft, ifft, fftfreq
from multiprocessing import Pool
from PIL import Image, ImageTk
from functools import partial
import tkinter as tk
from skimage.transform import resize
from skimage import img_as_ubyte
from tkinter import ttk, messagebox,filedialog
from pprint import pprint


def radon_transform(image, scan_count, detector_count, angle_range, pad=True, plot=False):
    if pad:
        image = rgb2gray(image)
        image = circle_pad(image)

    center = center_of(image)
    width = image.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, scan_count)
    results = np.zeros((scan_count, detector_count))

    if plot:
        plt.figure()
        for i, alpha in enumerate(alphas):
            results[i] = single_radon_transform(detector_count, angle_range, image, radius, center, alpha)
            plt.imshow(np.swapaxes(results, 0, 1), cmap=plt.cm.Greys_r)
    else:
        with Pool() as pool:
            results_list = pool.map(partial(single_radon_transform, detector_count, angle_range, image, radius, center),
                                    alphas)
        results = np.array(results_list)

    if plot:
        plt.imshow(np.swapaxes(results, 0, 1), cmap=plt.cm.Greys_r)

    return np.swapaxes(results, 0, 1)


def center_pad(array, shape, *args, **kwargs):
    pad = (np.array(shape) - np.array(array.shape)) / 2
    pad = np.array([np.floor(pad), np.ceil(pad)]).T.astype(int)
    return np.pad(array, pad, *args, **kwargs)


def circle_pad(array, *args, **kwargs):
    if len(array.shape) == 3:
        array = rgb2gray(array)
    shape = array.shape
    w, h = shape
    side = max(w, h)
    padded_array = center_pad(array, (side, side), *args, **kwargs)
    return padded_array

def center_of(array):
    return np.floor(np.array(array.shape) / 2).astype(int)


def rescale(array, min=0, max=1):
    res = array.astype('float32')
    res -= np.min(res)
    res /= np.max(res)
    res -= min
    res *= max
    return res


def clip(array, min, max):
    array[array < min] = min
    array[array > max] = max
    return array


def cut_pad(img, height, width):
    y, x = img.shape
    startx = x // 2 - (width // 2)
    starty = y // 2 - (height // 2)
    return img[starty:starty + height, startx:startx + width]


def bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) > abs(x1 - x0):
        swapped = True
        x0, y0, x1, y1 = y0, x0, y1, x1
    else:
        swapped = False
    m = (y1 - y0) / (x1 - x0) if x1 - x0 != 0 else 1
    q = y0 - m * x0
    if x0 < x1:
        xs = np.arange(np.floor(x0), np.ceil(x1) + 1, +1, dtype=int)
    else:
        xs = np.arange(np.ceil(x0), np.floor(x1) - 1, -1, dtype=int)
    ys = np.round(m * xs + q).astype(int)
    if swapped:
        xs, ys = ys, xs
    return np.array([xs, ys])


def radon_lines(emitters, detectors):
    return [np.array(bresenham(x0, y0, x1, y1)) for (x0, y0), (x1, y1) in zip(emitters, detectors)]


def single_radon_transform(detector_count, angle_range, image, radius, center, alpha):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = radon_lines(emitters, detectors)
    result = rescale(np.array([np.sum(image[tuple(line)]) for line in lines]))
    return result

def circle_points(angle_shift, angle_range, count, radius=1, center=(0, 0)):
    angles = np.linspace(0, angle_range, count) + angle_shift
    cx, cy = center
    x = radius * np.cos(angles) - cx
    y = radius * np.sin(angles) - cy
    points = np.array(list(zip(x, y)))
    return np.floor(points).astype(int)


def detector_coords(alpha, angle_range, count, radius=1, center=(0, 0)):
    return circle_points(np.radians(alpha - angle_range / 2), np.radians(angle_range), count, radius, center)


def emitter_coords(alpha, angle_range, count, radius=1, center=(0, 0)):
    return circle_points(np.radians(alpha - angle_range / 2 + 180), np.radians(angle_range), count, radius, center)[::-1]


def filter_sinogram(sinogram):
    n = sinogram.shape[0]
    filter = 2 * np.abs(fftfreq(n).reshape(-1, 1))
    result = ifft(fft(sinogram, axis=0) * filter, axis=0)
    result = clip(np.real(result), 0, 1)
    return result


def single_inverse_radon_transform(image, tmp, single_alpha_sinogram, alpha, detector_count, angle_range, radius,
                                   center):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = radon_lines(emitters, detectors)
    for i, line in enumerate(lines):
        image[tuple(line)] += single_alpha_sinogram[i]
        tmp[tuple(line)] += 1


def inverse_radon(shape, sinogram, angle_range, pad=True, plot=False, filtering=False):
    if filtering:
        sinogram = filter_sinogram(sinogram)
    number_of_detectors, number_of_scans = sinogram.shape
    sinogram = np.swapaxes(sinogram, 0, 1)

    result = np.zeros(shape)
    if pad:
        result = rgb2gray(result)
        result = circle_pad(result)
    tmp = np.zeros(result.shape)

    center = center_of(result)
    width = height = result.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, number_of_scans)

    for i, alpha in enumerate(alphas):
        single_inverse_radon_transform(result, tmp, sinogram[i], alpha, number_of_detectors, angle_range, radius,
                                       center)
        if plot:
            plt.imshow(result, cmap=plt.cm.Greys_r)

    tmp[tmp == 0] = 1
    result = rescale(result / tmp)
    result = (result * 255).astype(np.uint8)

    if pad:
        result = cut_pad(result, result.shape[0], result.shape[1])
    return result


def test(image, scans, detectors, angle_range, filtering, plot=False):
    padded = circle_pad(image)
    sinogram = radon_transform(padded, scans, detectors, angle_range, pad=False)
    output = inverse_radon(padded.shape, sinogram, angle_range, filtering=filtering, pad=False)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10,10))

        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(output, cmap='gray')
    output = resize(output, image.shape, anti_aliasing=True)

    if len(image.shape) == 3:
        image = rgb2gray(image)
    if len(output.shape) == 3:
        output = rgb2gray(output)
    res = rmse(image,output)
    return output,res

def filter_test(image, scans, detectors, angle_range):
    y1, loss1 = test(image, scans, detectors, angle_range, filtering=False)
    y2, loss2 = test(image, scans, detectors, angle_range, filtering=True)
    return rmse(y1,y2)




def read_dicom(path):
    from pydicom import dcmread
    ds = dcmread(path)
    keys = {x for x in dir(ds) if x[0].isupper()} - {'PixelData'}
    meta = {x: getattr(ds, x) for x in keys}
    image = ds.pixel_array
    return image, meta


def write_dicom(path, image, meta):
    from pydicom.dataset import Dataset, FileDataset, validate_file_meta
    from pydicom.uid import generate_uid
    from pydicom._storage_sopclass_uids import MRImageStorage

    ds = Dataset()
    ds.MediaStorageSOPClassUID = MRImageStorage
    ds.MediaStorageSOPInstanceUID = generate_uid()
    ds.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    fd = FileDataset(path, {}, file_meta=ds, preamble=b'\0' * 128)
    fd.is_little_endian = True
    fd.is_implicit_VR = False

    fd.SOPClassUID = MRImageStorage
    fd.PatientName = 'Test^Firstname'
    fd.PatientID = '123456'
    now = datetime.datetime.now()
    fd.StudyDate = now.strftime('%Y%m%d')

    fd.Modality = 'MR'
    fd.SeriesInstanceUID = generate_uid()
    fd.StudyInstanceUID = generate_uid()
    fd.FrameOfReferenceUID = generate_uid()

    fd.BitsStored = 16
    fd.BitsAllocated = 16
    fd.SamplesPerPixel = 1
    fd.HighBit = 15

    fd.ImagesInAcquisition = '1'
    fd.Rows = image.shape[0]
    fd.Columns = image.shape[1]
    fd.InstanceNumber = 1

    fd.ImagePositionPatient = r'0\0\1'
    fd.ImageOrientationPatient = r'1\0\0\0\-1\0'
    fd.ImageType = r'ORIGINAL\PRIMARY\AXIAL'

    fd.RescaleIntercept = '0'
    fd.RescaleSlope = '1'
    fd.PixelSpacing = r'1\1'
    fd.PhotometricInterpretation = 'MONOCHROME2'
    fd.PixelRepresentation = 1

    for key, value in meta.items():
        setattr(fd, key, value)

    validate_file_meta(fd.file_meta, enforce_standard=True)

    fd.PixelData = (image * 255).astype(np.uint16).tobytes()
    fd.save_as(path)

def rmse(a, b):
    a, b = rescale(a), rescale(b)
    return np.sqrt(np.mean((a - b)**2))

class TomographyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Tomography Simulator")

        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.pack(fill="both", expand=True)

        self.main_frame = ttk.Frame(self.scrollable_frame.scrollable_frame)
        self.main_frame.pack(padx=10, pady=10)

        self.input_image_frame = ttk.LabelFrame(self.main_frame, text="Input Image")
        self.input_image_frame.grid(row=0, column=0, padx=10, pady=10)

        self.default_image = None
        self.input_image = None
        self.input_image_label = ttk.Label(self.input_image_frame)
        self.input_image_label.pack()

        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.controls_frame.grid(row=0, column=1, padx=10, pady=10)

        self.load_button = ttk.Button(self.controls_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(padx=10, pady=5)
        self.load_dicom_button = ttk.Button(self.controls_frame, text="Load DICOM", command=self.load_dicom)
        self.load_dicom_button.pack(padx=10, pady=5)
        self.transform_button = ttk.Button(self.controls_frame, text="Transform", command=self.perform_transform)
        self.transform_button.pack(padx=10, pady=5)
        self.reset_button = ttk.Button(self.controls_frame, text="Reset", command=self.reset_image)
        self.reset_button.pack(padx=10, pady=5)
        self.save_button = ttk.Button(self.controls_frame, text="Save DICOM", command=self.save_dicom)
        self.save_button.pack(padx=10, pady=5)

        self.filtering_var = tk.BooleanVar()
        self.filtering_var.set(True)
        self.filtering_checkbox = ttk.Checkbutton(self.controls_frame, text="Enable Filtering", variable=self.filtering_var)
        self.filtering_checkbox.pack(padx=10, pady=5)

        self.test_button = ttk.Button(self.controls_frame, text="Test", command=self.perform_test)
        self.test_button.pack(padx=10, pady=5)
        self.test_filtering_button = ttk.Button(self.controls_frame, text="Test Filtering", command=self.perform_test_filtering)
        self.test_filtering_button.pack(padx=10, pady=5)

        self.output_image_frame = ttk.LabelFrame(self.main_frame, text="Output Image")
        self.output_image_frame.grid(row=0, column=2, padx=10, pady=10)

        self.output_image = None
        self.output_image_label = ttk.Label(self.output_image_frame)
        self.output_image_label.pack()

        self.sinogram_frame = ttk.LabelFrame(self.main_frame, text="Sinogram")
        self.sinogram_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        self.sinogram_image = None
        self.sinogram_image_label = ttk.Label(self.sinogram_frame)
        self.sinogram_image_label.pack()

        self.patient_info_frame = ttk.LabelFrame(self.main_frame, text="Patient Information")
        self.patient_info_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.patient_name_label = ttk.Label(self.patient_info_frame, text="Patient Name:")
        self.patient_name_label.grid(row=0, column=0, padx=5, pady=5)
        self.patient_name_entry = ttk.Entry(self.patient_info_frame)
        self.patient_name_entry.grid(row=0, column=1, padx=5, pady=5)

        self.patient_id_label = ttk.Label(self.patient_info_frame, text="Patient ID:")
        self.patient_id_label.grid(row=1, column=0, padx=5, pady=5)
        self.patient_id_entry = ttk.Entry(self.patient_info_frame)
        self.patient_id_entry.grid(row=1, column=1, padx=5, pady=5)

        self.image_comments_label = ttk.Label(self.patient_info_frame, text="Image Comments:")
        self.image_comments_label.grid(row=2, column=0, padx=5, pady=5)
        self.image_comments_entry = ttk.Entry(self.patient_info_frame)
        self.image_comments_entry.grid(row=2, column=1, padx=5, pady=5)

        self.study_date_label = ttk.Label(self.patient_info_frame, text="Study Date:")
        self.study_date_label.grid(row=3, column=0, padx=5, pady=5)
        self.study_date_entry = ttk.Entry(self.patient_info_frame)
        self.study_date_entry.grid(row=3, column=1, padx=5, pady=5)

        self.dicom_info_frame = ttk.LabelFrame(self.main_frame, text="DICOM Image and Info")
        self.dicom_info_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        self.dicom_image_label = ttk.Label(self.dicom_info_frame)
        self.dicom_image_label.grid(row=0, column=0, padx=5, pady=5)

        self.dicom_info_text = tk.Text(self.dicom_info_frame, height=10, width=50)
        self.dicom_info_text.grid(row=0, column=1, padx=5, pady=5)

        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.controls_frame.grid(row=4, column=1, padx=10, pady=10)

        self.delta_alpha_label = ttk.Label(self.controls_frame, text="Step (∆α) for emitter/detector arrangement:")
        self.delta_alpha_label.pack(padx=10, pady=5)
        self.delta_alpha_entry = ttk.Entry(self.controls_frame)
        self.delta_alpha_entry.pack(padx=10, pady=5)

        self.detector_count_label = ttk.Label(self.controls_frame, text="Number of detectors (n) for one arrangement:")
        self.detector_count_label.pack(padx=10, pady=5)
        self.detector_count_entry = ttk.Entry(self.controls_frame)
        self.detector_count_entry.pack(padx=10, pady=5)

        self.angle_range_label = ttk.Label(self.controls_frame,
                                           text="Angular range (l) for emitter/detector arrangement:")
        self.angle_range_label.pack(padx=10, pady=5)
        self.angle_range_entry = ttk.Entry(self.controls_frame)
        self.angle_range_entry.pack(padx=10, pady=5)

    def perform_test(self):
        if self.default_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        try:
            output,loss = test(self.default_image, 360, 360, 270, self.filtering_var.get())
            messagebox.showinfo("Test Result", f"RMSE: {loss:.6f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def perform_test_filtering(self):
        if self.default_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        filtering = self.filtering_var.get()

        try:
            loss = filter_test(self.default_image, 360, 360, 270)
            messagebox.showinfo("Filtering Test Result", f"RMSE: {loss:.6f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_dicom(self):
        file_path = filedialog.askopenfilename(title="Select DICOM Image", filetypes=[("DICOM files", "*.dcm")])
        if file_path:
            try:
                image, meta = read_dicom(file_path)

                self.dicom_info_text.delete(1.0, tk.END)
                for key, value in meta.items():
                    self.dicom_info_text.insert(tk.END, f"{key}: {value}\n")

                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                self.dicom_image_label.config(image=image)
                self.dicom_image_label.image = image

                if meta:
                    self.patient_name_entry.delete(0, tk.END)
                    self.patient_name_entry.insert(0, meta.get('PatientName', ''))

                    self.patient_id_entry.delete(0, tk.END)
                    self.patient_id_entry.insert(0, meta.get('PatientID', ''))

                    self.image_comments_entry.delete(0, tk.END)
                    self.image_comments_entry.insert(0, meta.get('ImageComments', ''))

                    self.study_date_entry.delete(0, tk.END)
                    self.study_date_entry.insert(0, meta.get('StudyDate', ''))
            except Exception as e:
                messagebox.showerror("Error", str(e))


    def save_dicom(self):
        if self.default_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        if self.output_image is None:
            messagebox.showerror("Error", "No processed image available.")
            return

        try:
            output_image_rescaled = rescale(self.output_image)
            padded_output_image = circle_pad(output_image_rescaled)

            save_path = filedialog.asksaveasfilename(
                title="Save DICOM File",
                filetypes=[("DICOM files", "*.dcm")],
                defaultextension=".dcm"
            )

            if save_path:
                patient_name = self.patient_name_entry.get()
                patient_id = self.patient_id_entry.get()
                image_comments = self.image_comments_entry.get()
                study_date = self.study_date_entry.get()

                meta = {
                    'PatientName': patient_name,
                    'PatientID': patient_id,
                    'ImageComments': image_comments,
                    'StudyDate': study_date,
                }

                write_dicom(save_path, padded_output_image, meta)
                messagebox.showinfo("Success", "DICOM file saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                self.default_image = imread(file_path)

                if self.default_image.dtype == np.uint8:
                    self.default_image = self.default_image.astype(np.float64) / 255.0

                max_height = 400
                max_width = 400
                img_height, img_width = self.default_image.shape[:2]
                if img_height > max_height or img_width > max_width:
                    scale_factor = min(max_height / img_height, max_width / img_width)
                    new_height = int(img_height * scale_factor)
                    new_width = int(img_width * scale_factor)
                    self.default_image = resize(self.default_image, (new_height, new_width))

                if self.default_image.dtype == np.float64:
                    self.default_image = img_as_ubyte(self.default_image)

                self.input_image = ImageTk.PhotoImage(image=Image.fromarray(self.default_image))
                self.input_image_label.config(image=self.input_image)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def perform_transform(self):
        if self.default_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return

        filtering = self.filtering_var.get()

        try:
            try:
                delta_alpha = int(self.delta_alpha_entry.get())
            except ValueError:
                delta_alpha = 360

            try:
                detector_count = int(self.detector_count_entry.get())
            except ValueError:
                detector_count = 360

            try:
                angle_range = int(self.angle_range_entry.get())
            except ValueError:
                angle_range = 270

            sinogram = radon_transform(self.default_image, delta_alpha, detector_count, angle_range, plot=False)
            if filtering:
                sinogram_filtered = filter_sinogram(sinogram)
                sinogram_rescaled = rescale(sinogram_filtered, min=0, max=255).astype(np.uint8)
            else:
                sinogram_rescaled = rescale(sinogram, min=0, max=255).astype(np.uint8)
            sinogram_image = ImageTk.PhotoImage(image=Image.fromarray(sinogram_rescaled))
            self.sinogram_image_label.config(image=sinogram_image)
            self.sinogram_image_label.image = sinogram_image
        except Exception as e:
            messagebox.showerror("Error radon", str(e))
            return

        try:
            if filtering:
                output_image = inverse_radon(self.default_image.shape, sinogram_filtered, angle_range, filtering=True)
            else:
                output_image = inverse_radon(self.default_image.shape, sinogram, angle_range, filtering=False)
            output_image_rescaled = rescale(output_image, min=0, max=255).astype(np.uint8)
            padded_output_image = circle_pad(output_image_rescaled)
            output_image_tk = ImageTk.PhotoImage(image=Image.fromarray(padded_output_image))
            self.output_image = output_image
            self.output_image_label.config(image=output_image_tk)
            self.output_image_label.image = output_image_tk
        except Exception as e:
            messagebox.showerror("Error inverse", str(e))
            return


    def reset_image(self):

        self.input_image_label.config(image=None)
        self.input_image = None

        self.output_image_label.config(image=None)
        self.output_image = None

        self.sinogram_image_label.config(image=None)
        self.sinogram_image = None

        self.default_image = None

        self.dicom_image_label.config(image=None)
        self.dicom_info_text.delete(1.0, tk.END)
        self.patient_name_entry.delete(0, tk.END)
        self.patient_id_entry.delete(0, tk.END)
        self.image_comments_entry.delete(0, tk.END)
        self.study_date_entry.delete(0, tk.END)
        self.delta_alpha_entry.delete(0, tk.END)
        self.detector_count_entry.delete(0, tk.END)
        self.angle_range_entry.delete(0, tk.END)

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


def main():
    app = TomographyApp()
    app.mainloop()

if __name__ == "__main__":
    main()
