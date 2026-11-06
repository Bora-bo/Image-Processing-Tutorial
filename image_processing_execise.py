import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    # importing necessary libraries for this exercise
    import marimo as mo
    import pandas as pd
    import numpy as np
    import cv2 as cv
    import matplotlib.pylab as plt

    from pathlib import Path
    import re
    import warnings # lets silence the warning
    warnings.filterwarnings('ignore')


    print('All import successful')
    return Path, cv, mo, np, plt, re


@app.cell
def _():
    #png_files = Path(".").rglob("*.png")

    #for file_path in png_files:
    #    print(file_path.name)        # File name with extension: "photo.png"
    #    print(file_path.stem)        # File name without extension: "photo"
    #    print(file_path.suffix)      # Extension only: ".png"
    #    print(file_path.parent)      # Directory path: "images"
    #    print(file_path.exists())    # Check if file exists: True
    return


@app.cell
def _(Path, re):
    # Getting image data from file location using Paths
    cat_pics = list(Path(r'C:\Users\user\Documents\Image processing exercise\Datasets\Tutorial datasets\training_set\cats').rglob('*.jpg'))

    dog_pics = list(Path(r'C:\Users\user\Documents\Image processing exercise\Datasets\Tutorial datasets\training_set\dogs').rglob('*.jpg'))

    def extract_numbers(path):
        '''Reorganise files by the image number rather than alphabetically'''
        sorted_paths = re.findall(r'\d+', path.name)
        return int(sorted_paths[0]) if sorted_paths else 0

    # The function is called in sorted()
    cat_photos = sorted(cat_pics, key=extract_numbers)
    dog_photos = sorted(dog_pics, key=extract_numbers)
    return cat_photos, dog_photos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Paths are sorted in alphabetical order. The extract_numbers function was used to sort the paths using the file post-fix number as the sorting order
    """)
    return


@app.cell
def _(cat_photos, cv, plt):
    # Using matplotlib and opencv .imread() to read image data
    mpl_read = plt.imread(cat_photos[20])
    cv_read = cv.imread(cat_photos[20])
    return cv_read, mpl_read


@app.cell
def _(cv_read, mpl_read):
    print(mpl_read.shape, cv_read.shape)
    return


@app.cell
def _(mpl_read, plt):
    # Displaying an image with imshow()
    # This image was imported via matplotlib

    fig, ax = plt.subplots(figsize = (5, 5))
    ax.imshow(mpl_read)
    ax.axis('off')
    plt.title('Read in with matplotlib')
    plt.show()
    return


@app.cell
def _(cv_read, plt):
    # This image was imported via opencv

    fig1, ax1 = plt.subplots(figsize = (5, 5))
    ax1.imshow(cv_read)
    ax1.axis('off')
    plt.title('Read in with opencv')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Matplotlib and open cv read in images differently. Specifically, they read in the image channels (Red, Green & Blue: (RGB)) in different orders.<br>Matplotlib reads: RGB <br>Opencv reads: BGR**
    """)
    return


@app.cell
def _(cv_read, mpl_read, plt):
    # comparing images side by side
    def compare_images(img, img1, title):
        fig1, ax1 = plt.subplots(1, 2, figsize = (10, 8), constrained_layout=True)

        ax1[0].imshow(img)
        ax1[1].imshow(img1)
        ax1[0].axis('off')
        ax1[1].axis('off')

        fig1.suptitle(title, y = 0.91)
        plt.show()

    compare_images(mpl_read, cv_read, 'Comparing matplotlib and opencv read images')
    return


@app.cell
def _(cv, cv_read, plt):
    # Reorder opencv BGR to RGB
    img_rgb = cv.cvtColor(cv_read, cv.COLOR_BGR2RGB) #takes the image data, not the image itself

    fig2, ax2 = plt.subplots(figsize = (5,5))
    ax2.imshow(img_rgb)
    ax2.axis('off')
    ax2.set_title('Converting BGR to RGB')

    plt.show()


    # numpy number indexing can also be used to reverse the colors - img_rgb = img_bgr[:, :, ::-1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualising the RGB channels
    <div class = alert alertblock alertinfo>
    <span style = 'color:yellowgreen; font-size:16px'> A color image has 3 dimensions: In the first channel is the length, in the second is the width, and in the third are the colors channels.<br>By manipulating the values of the 3rd dimension, the different color channels can be observed.</span>
    </div>
    """)
    return


@app.function
def display_image(img, channel, color, ax3):
    '''Display read data into images'''
    ax3.imshow(img[:,:,channel])
    ax3.axis('off')
    ax3.set_title(f'{color} channel')


@app.cell
def _(mpl_read, plt):
    fig4, ax4 = plt.subplots(1,3, figsize = (15,15))
    display_image(mpl_read, 0, 'Red', ax4[0]) #toggling the red channel only
    display_image(mpl_read, 1, 'Green', ax4[1]) #toggling the green channel only
    display_image(mpl_read, 2, 'Blue', ax4[2]) #toggling the blue channel only

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the process of separating out the channel, if cmap is unspecified (cmap is a parameter in plt.imshow), the the default coloring scheme - viridis - is used. This ranges from 'dark purple' for low intensity colors to 'light yellow-green' for high intensity colors.**

    **Setting an image cmap to 'Grays' will result in a greyscale image.**
    """)
    return


@app.cell
def _(mpl_read, plt):
    # An alternate way to display images with loops rather than functions
    fig5, ax5 = plt.subplots(1, 3, figsize=(15, 15))
    channels = ['Red', 'Green', 'Blue']
    color_map = ['Reds', 'Greens', 'Blues']

    for _i, _col in enumerate(channels):
        ax5[_i].imshow(mpl_read[:, :, _i], cmap = color_map[_i])
        ax5[_i].set_title(f'{_col} Channel')
        ax5[_i].axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mpl_read, plt):
    # Converting image to greyscale
    plt.imshow(mpl_read[:,:,1], cmap = 'Grays')

    plt.axis('off')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Sometimes images come with a 4th channel, alpha (which controls pixel transparency (or opacity)).**

    **Matplotlib natively reads the 4th channel and applies pixel-level transparency/blending when rendering.**

    **OpenCv by default, functions like cv2.imread() drop the alpha channel entirely. To preserve it, you must explicitly use the cv2.IMREAD_UNCHANGED flag.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### View an image with Red, Green and Blue colors present to better understand the channels
    """)
    return


@app.cell
def _(dog_photos, plt):
    colorch_exp = plt.imread(dog_photos[2279]) #reading image

    plt.imshow(colorch_exp)
    plt.axis('off')
    plt.show()
    return (colorch_exp,)


@app.cell
def _(colorch_exp, plt):
    fig6, ax6 = plt.subplots(1, 3, figsize=(15,5))

    channel_colors = ['Reds', 'Greens', 'Blues']
    interest_areas = ['Observe the red car and reddish scarf', 'Observe the grass', 'Observe the blue strap',]

    for _i, _col in enumerate(channel_colors):
        ax6[_i].imshow(colorch_exp[:,:,_i], cmap = _col)
        ax6[_i].set_title(interest_areas[_i])
        ax6[_i].axis('off')

    plt.show()
    return (channel_colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The colors we see are the reflection of the dominant wavelength being reflected back at us.**


    **The blue strap, while the strap is light in the blue channel (because the strap is blue). It does not appear as light as the red car in the red channel or the green grass in the green channel.**

    **The uneven reflection of the car in the red channel could be the result of unequal lighting. The scafe could be the result of too much exposure to the sun.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class = alert alert-block alert-info>
        <span style='color:lightpink; font-size:18px'> To better understand the 3 channels a picture with uniform lighting, containing all 3 colors would be used.</span>
    </div>
    """)
    return


@app.cell
def _(Path):
    new_obj = list(Path(r'C:\Users\user\Documents\Image processing exercise\Datasets\Tutorial datasets').rglob('Red_eyed_tree_frog_example_image_to_examine_img_channels.jpg'))
    return (new_obj,)


@app.cell
def _(new_obj, plt):
    frog_img = plt.imread(new_obj[0])
    return (frog_img,)


@app.cell
def _(frog_img, plt):
    plt.imshow(frog_img)

    plt.axis('off')
    plt.show()
    return


@app.cell
def _(channel_colors, frog_img, plt):
    fig_frog, ax_frog = plt.subplots(1,3, figsize=(12,6))

    for _i, _col in enumerate(channel_colors):
        ax_frog[_i].imshow(frog_img[:,:,_i], cmap = _col)
        ax_frog[_i].axis('off')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resizing and rescaling
    """)
    return


@app.cell
def _(dog_photos, plt):
    img_file = plt.imread(dog_photos[881])

    photo_file = plt.imshow(img_file) #displaying image
    plt.axis('off')

    plt.show()
    return (img_file,)


@app.cell
def _(cv, img_file, plt):
    # Both resize parameters -dsize and fx, fy cannot be null or zero to prevent an error

    _img_resized = cv.resize(img_file, dst=None, dsize=None, fx=0.3, fy=0.5)

    plt.imshow(_img_resized)
    plt.axis('off')

    plt.show()
    return


@app.cell
def _(cv, img_file, plt):
    # Both resize parameters -dsize and fx, fy cannot be null or zero to prevent an error

    _img_resized = cv.resize(img_file, dst=None, dsize=(1000,1000), fx=0, fy=0)

    plt.imshow(_img_resized)
    plt.axis('off')

    plt.show()
    return


@app.cell
def _(cv, img_file, plt):
    _img_resized = cv.resize(img_file, dst=None, dsize=(1000,1000), fx=0, fy=0, interpolation = cv.INTER_CUBIC)

    plt.imshow(_img_resized)
    plt.axis('off')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image kernels 🎦

    **In image processing, a kernel (often called a convolution matrix or mask) is a small matrix of numbers used to perform image filtering operations.**

    **By sliding the kernel across an image and applying a mathematical operation called convolution (cross-correlation, depending of the package) you can modify pixel values to achieve various effects, such as blurring, sharpening, edge detection, and noise reduction.**

    **Note:**
    **Kernels are also used for correlation image operations but unlike in convolution where values are flipped, correlation does not flip the values.**

    **It can be used for image pattern matching.**
    """)
    return


@app.cell
def _(cat_photos, cv, plt):
    # Using numpy array defined kernels to apply filters 
    # Fetch image 

    wrkImage = cv.imread(cat_photos[2322])

    plt.imshow(wrkImage)
    plt.axis('off')

    plt.show()
    return (wrkImage,)


@app.cell
def _(wrkImage):
    print(wrkImage.shape)
    return


@app.cell
def _(cv, np, plt, wrkImage):
    # Sharpen image

    # Define a kernel with np.array and apply it to the above image using cv filter2D

    Img_filter = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])

    sharpened_img = cv.filter2D(src = wrkImage, ddepth = -1, kernel = Img_filter, borderType = cv.BORDER_REFLECT)
    plt.imshow(sharpened_img)
    plt.axis('off')
    plt.show()
    return Img_filter, sharpened_img


@app.cell
def _(sharpened_img):
    print(sharpened_img.shape)
    return


@app.cell
def _(cv, np, plt, wrkImage):
    Img_filter1 = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])
        #the dot product of the kernel is 0. The original pixel values of image darkens (the image value multiplied by the the kernel coefficients,0, move the color towards darker intensities)

    sharpened_img1 = cv.filter2D(src = wrkImage, ddepth = -1, kernel = Img_filter1, borderType = cv.BORDER_REFLECT)
    plt.imshow(sharpened_img1)
    plt.axis('off')
    plt.show()
    return (sharpened_img1,)


@app.cell
def _(cv, np, plt, wrkImage):
    # Applying a blur filter

    blurred_filter = np.ones((3,3), np.float32)/9

    # Applying blurred filter to image
    blurred_image = cv.filter2D(wrkImage, -1, blurred_filter)
    plt.imshow(blurred_image)
    plt.axis('off')
    plt.show()
    return (blurred_image,)


@app.cell
def _(Img_filter, blurred_image, cv, plt):
    # Sharpening the blurred image
    sharpe_img = cv.filter2D(src = blurred_image, ddepth = -1, kernel = Img_filter, borderType = cv.BORDER_REFLECT)
    #new_image = plt.imshow(sharpe_img)

    plt.imshow(sharpe_img)
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(plt, sharpened_img1, wrkImage):
    # Sharpening the blurred image
    new_image = sharpened_img1 + wrkImage

    plt.imshow(new_image)
    plt.axis('off')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The noise in the image is likely due to clipping.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class = alert alert-block alert-class>
        <span style = 'color:lightpink; font-size:19px'>Interestingly, dividing by a lower number brightens the image while dividing by a larger number darkens the image. Is behaviour is due to the sum of kernel coefficients, dividing by smaller values leads to sums greater than 1 while larger divisors lead to a lower summed value. <br>
        Dividing by the sum of all coefficients allow the filter to blur the image without altering the brightness.<br>
        </span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Selecting regions in an image
    """)
    return


@app.cell
def _(cat_photos, cv):
    img_var = cv.imread(cat_photos[3000])
    cutie_data = cv.cvtColor(img_var, cv.COLOR_BGR2RGB) #setting color order to RGB
    return (cutie_data,)


@app.cell
def _(cutie_data, plt):
    cutie_pie = plt.imshow(cutie_data)
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(cutie_data, plt):
    plt.imshow(cutie_data[150:300, 10:200]) #slicing the image
    plt.axis('off')

    plt.show()
    return


@app.cell
def _(cutie_data, plt):
    fig_cat, ax_cat = plt.subplots(1,3, figsize=(17,10))
    two_channels = ['G+B = Cyan', 'R+B = Purple', 'R+G = Yellow']

    for _i in range(3):
        cat_data_copied = cutie_data.copy()
        cat_data_copied[:,:,_i] = 0
        ax_cat[_i].imshow(cat_data_copied) #setting each channel to 0
        ax_cat[_i].axis('off')
        ax_cat[_i].set_title(two_channels[_i])


    plt.show()
    return


@app.cell
def _(cutie_data, cv):
    # Checking the speed in (seconds) of a cv operation with getTickCount
    import time

    sample_img = cutie_data.copy()

    tt1 = time.time()
    e1 = cv.getTickCount()

    for _i in range(5,49,2):
        img1 = cv.medianBlur(sample_img,_i)

    e2 = cv.getTickCount()

    t = (e2 - e1)/cv.getTickFrequency()
    tt2 = time.time()

    print(tt2 - tt1)
    print( t )
    print(e1)
    print(e2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **getTickCount() count the number of clock cycles (ticks) from the systems oscillator (device on the motherboard)**

    **On its own the value is not very useful. However, as used above, e2 - e1 will give you the number of ticks for which a function ran.**

    **A computers clock cycle can go for billions of ticks per second hence to human readable time t (which is e2 - e1) is divided by getTickFrequency().**

    **The function time() does the same without the need to divide by getTickFrequency().**
    """)
    return


@app.cell
def _(cv):
    cv.useOptimized()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Changing colorspace with opencv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class = alert alert-block alert-class>
        <span style = 'color:lightpink; font-size:18px'>The function <b>cv.cvtColor(input_img, flag)</b> is used to convert the color of an image. <br> The flag parameter determines the type of conversion.</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How many flags are there?**
    """)
    return


@app.cell
def _(cv):
    # Printing out all flags

    flags = [_i for _i in dir(cv) if _i.startswith('COLOR_')]
    print( flags[:6] )
    print(len(flags))

    # Iterating through the list 'flags'
    #for _i in flags:
    #    print(_i)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class = alert alert-block alert-class>
        <span style = 'color:lightpink; font-size:17px'>For HSV, <br><b>Hue</b> range is [0,179], </br><b>Saturation</b> range is [0,255], and <br><b>Value</b> range is [0,255]. <br>Different software use different scales. So if you are comparing OpenCV values with them, you need to normalize these ranges.</span></div>
    """)
    return


@app.cell(hide_code=True)
def _(cat_photos, cv):
    bgr_cat = cv.imread(cat_photos[2000])
    #print(hsv_cat)
    return (bgr_cat,)


@app.cell
def _(bgr_cat, plt):
    plt.imshow(bgr_cat)
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(bgr_cat, cv, plt):
    # Covert BGR colorspace to RGB
    rgb_cat = cv.cvtColor(bgr_cat, cv.COLOR_BGR2RGB)

    plt.imshow(rgb_cat)
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(bgr_cat, cv, plt):
    # Covert to greyscale
    grey_cat = cv.cvtColor(bgr_cat, cv.COLOR_BGR2GRAY)

    plt.imshow(grey_cat, cmap='gray', vmin=0, vmax=255) 
    #use colormap when displaying with matplotlib otherwise it will map to its defalut folor

    plt.axis('off')
    plt.colorbar()
    plt.show()
    return (grey_cat,)


@app.cell
def _(grey_cat):
    # Checking grey scale image dimensions
    print(grey_cat.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The 3 dimensional image is now 2 dimensions. Color channels has been removed.**
    """)
    return


@app.cell
def _(bgr_cat, cv, plt):
    # Covert to HSV
    hsv_cat = cv.cvtColor(bgr_cat, cv.COLOR_BGR2HSV)

    plt.imshow(hsv_cat)
    plt.axis('off')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The blocky colour distortions in the HSV channels are mainly JPEG compression artefacts. JPEG stores colour at a lower resolution than brightness and adds small rounding errors to each block of pixels, which mostly damages the hue and saturation channels.
    - This image contains almost no colour. Its saturation channel (how far each pixel’s colour is from gray) is close to zero nearly everywhere, so the hue there is essentially arbitrary: when R, G and B are almost equal, a difference of 1 or 2 can change which channel is largest and flip the hue to a completely different colour. Hue also wraps around the colour wheel, so red appears at both ends of OpenCV’s 0–179 range.
    - Hue is a reliable way to identify the colours in an image or region, but only where saturation and brightness (V) are high enough. This makes it useful for tasks such as judging fruit ripeness, reading traffic lights, or analysing stained cells.
    - Because a change in brightness mainly affects V, hue is largely unaffected by lighting intensity, which makes it more suitable than raw RGB values for many colour-based tasks.
    - Ways to reduce these problems, such as masking out low-saturation and low-brightness pixels, will be covered later.
    """)
    return


@app.cell
def _(bgr_cat, cv, plt):
    _hsv = cv.cvtColor(bgr_cat, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(_hsv)

    fig_hsv, axes_hsv = plt.subplots(1, 3, figsize=(15, 5))
    axes_hsv[0].imshow(h, cmap='hsv', vmin=0, vmax=179); axes_hsv[0].set_title('Hue')
    axes_hsv[1].imshow(s, cmap='viridis', vmin=0, vmax=255); axes_hsv[1].set_title('Saturation')
    axes_hsv[2].imshow(v, cmap='gray', vmin=0, vmax=255); axes_hsv[2].set_title('Value')
    axes_hsv[0].axis('off')
    axes_hsv[1].axis('off')
    axes_hsv[2].axis('off')
    plt.show()
    return


@app.cell
def _(cv, dog_photos, plt):
    anth_cat_info = cv.imread(dog_photos[545])

    anth_cat_cvt = cv.cvtColor(anth_cat_info, cv.COLOR_BGR2RGB)


    plt.imshow(anth_cat_cvt)
    plt.axis('off')
    plt.show()
    return (anth_cat_cvt,)


@app.cell
def _(anth_cat_cvt, cv, plt):
    now_dog = cv.cvtColor(anth_cat_cvt, cv.COLOR_RGB2HSV)
    _colormap = ['hsv', 'viridis', 'gray']

    dog_fig, dog_axes = plt.subplots(1, 3, figsize=(10,15))

    for _i in range(3):
        dog_axes[_i].imshow(now_dog[:,:,_i], _colormap[_i])
        dog_axes[_i].axis('off')


    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - HSV works well on photos with plenty of real colour, and poorly on images made mostly of blacks, whites and grays (like the cat image above). Even in a good image, the dark and gray areas (like the dog’s fur) have unreliable hue, so you still need to mask out pixels with low saturation or low V before using hue.
    """)
    return


@app.cell
def _():
    # # Picking up pink/red objects in camera frame
    # cap = cv.VideoCapture(0)
    # try:
    #     while True:
    #         ok, frame = cap.read()
    #         if not ok:                      # camera failed, so stop instead of crashing
    #             break

    #         lower_pink1 = np.array([145, 40, 120])   # magenta-pink side
    #         upper_pink1 = np.array([179, 255, 255])
    #         lower_pink2 = np.array([0, 40, 120])     # the red-pink side, after the wrap
    #         upper_pink2 = np.array([5, 255, 255])

    #         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #         mask = cv.inRange(hsv, lower_pink1, upper_pink1) | cv.inRange(hsv, lower_pink2, upper_pink2)
    #         res = cv.bitwise_and(frame, frame, mask=mask)

    #         cv.imshow('frame', frame)
    #         cv.imshow('mask', mask)
    #         cv.imshow('res', res)

    #         k = cv.waitKey(5) & 0xFF
    #         if k in (27, ord('q')):         # Esc or q, with an OpenCV window focused
    #             break
    # finally:
    #     cap.release()                       # always free the camera
    #     cv.destroyAllWindows()
    #     cv.waitKey(1)                       # lets some systems (macOS especially) actually close the windows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Range of blue colors <br>
    lower_blue = np.array([110,50,50]) <br>
    upper_blue = np.array([130,255,255])
    <br><br>
    - Depicting colors with BGR
    green = np.uint8([[[0,255,0 ]]])<br>
    hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)<br>
    print( hsv_green )<br>
    [[[ 60 255 255]]]<br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometric Transformation of images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## cv.resize()
    """)
    return


@app.cell
def _(plt):
    # Let's create a function to display images with matplotlib without the axis

    def display(_img):
        plt.imshow(_img)
        plt.axis('off')
        plt.show()

    return (display,)


@app.cell
def _(cv, dog_photos, plt):
    exp_dimg = cv.imread(dog_photos[1111])
    cc_dimg = cv.cvtColor(exp_dimg, cv.COLOR_BGR2RGB)

    plt.imshow(cc_dimg)
    plt.axis('off')
    plt.show()
    return (cc_dimg,)


@app.cell
def _(cc_dimg, cv, display):
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    res = cv.resize(src = cc_dimg, dsize = None, fx=2, fy=2, interpolation = cv.INTER_CUBIC)
    #other interpolation parameters include: cv.INTER_LINEAR(for zooming), cv.INTER_AREA(for shrinking)
    display(res)
    return (res,)


@app.cell
def _(cc_dimg, res):
    # cv.resize() was used to rescale the image by a factor of 2
    print(f'Original dimensions: {cc_dimg.shape}')
    print(f'Rescaled dimensions: {res.shape}')
    return


@app.cell
def _(cc_dimg, plt, res):
    # comparing the original and scaled images side by side
    _imgFig, _imgAx = plt.subplots(1,2, figsize = (9,12))

    _imgAx[0].imshow(cc_dimg)
    _imgAx[1].imshow(res)
    _imgAx[0].axis('off')
    _imgAx[1].axis('off')
    _imgAx[0].set_title('Original')
    _imgAx[1].set_title('Scaled')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Scaling the image down results in loss of information. As a result the image become blurry and pixelated.
    - An image can never look more detailed than its original source data. No matter the value used to scale up an image this fact would always be true, because once the information ceiling is reached it cannot be passed.


    For more on how to rescale images and get additional information that normal resize can't give look into **AI Super-Resolution**

    **cv2.dnn_superres (Deep Neural Network Super-Resolution).**
    """)
    return


@app.cell
def _(cc_dimg, cols, cv, display, np, rows):
    # the gray_dog variable was an attempt of this transformation with the image in greyscale
    #grey_dog = cv.cvtColor(cc_dimg, cv.COLOR_RGB2GRAY)
    _rows,_cols,_chl = cc_dimg.shape

    _M = np.float32([[1,0,100],[0,1,50]])
    _dst = cv.warpAffine(src=cc_dimg, dst=None, M=_M, dsize=(cols,rows))

    display(_dst)

    # cv.imshow('img',dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - This transformation has some really cool uses. Trust me.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotation Transformation
    """)
    return


@app.cell
def _(cc_dimg, cv, display):
    grey_dog = cv.cvtColor(cc_dimg, cv.COLOR_RGB2GRAY)
    _rows,_cols = grey_dog.shape

    # Appling a 90 degree transformation
    ## cols-1 and rows-1 are the coordinate limits.
    _M = cv.getRotationMatrix2D(((_cols-1)/2.0,(_rows-1)/2.0),90,1)
    _dst = cv.warpAffine(grey_dog, _M,(_cols,_rows))
    # It appears cv.wrapAffine applies whatever transformation is applied to M

    display(_dst)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perspective Transform
    """)
    return


@app.cell
def _(cc_dimg, cv, np, plt):
    _pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    _pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    _M = cv.getPerspectiveTransform(_pts1,_pts2)

    _dst = cv.warpPerspective(cc_dimg,_M,(300,300))

    plt.subplot(121),plt.imshow(cc_dimg),plt.axis('off'),plt.title('Input')
    plt.subplot(122),plt.imshow(_dst),plt.axis('off'),plt.title('Output')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
