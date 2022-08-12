from importlib.machinery import WindowsRegistryFinder
from importlib.resources import path
from sys import float_info
from scipy import spatial
import string
import random
import numpy as np
from ipywidgets import widgets
import traitlets
from typing import Tuple
import random_word
import pathlib
import matplotlib.patches as patches
import matplotlib as mpl

WORDS = []
compounds = ["CZX-1", "ZnCl2", "CuCl", "halozeotype", "amine", "hydrate"]


def word_pair():
    global WORDS
    if not WORDS:
        p = pathlib.Path(__file__).absolute().parent.parent / "english-adjectives.txt"
        WORDS = open(p, "r").read().split("\n")
    return f"{random.choice(WORDS)} {random.choice(compounds)}"


class Point(widgets.Text):
    x = traitlets.Float()
    y = traitlets.Float()

    def __init__(self, x, y, *args, **kwargs):
        super(Point, self).__init__(continuous_update=False, *args, **kwargs)
        self.x = x
        self.y = y
        self.observe(self._update, names=["x", "y", "value"])
        self.value = f"{x},{y}"

    def _update(self, change):
        print(change)
        if change["name"] == "value":
            x, y = self.value.split(",")
            with self.hold_trait_notifications():
                self.x = float(x)
                self.y = float(y)
        if change["name"] == "x" or change["name"] == "y":
            self.value = f"{self.x},{self.y}"


class Point_try2(widgets.HBox):
    x = traitlets.Float()
    y = traitlets.Float()

    def __init__(self, x, y, *args, **kwargs):
        super(Point, self).__init__(*args, **kwargs)
        # self.x = x
        # self.y = y
        self.text_x = widgets.Text()
        self.text_y = widgets.Text()
        # for obj in [self.x, self.y, self.text_x, self.text_y]:
        #     obj.observe(self._update)
        # self.text_x.observe(self._update, names=['value'])
        self.observe(self._update, names=["x", "y"])
        self.text_x.observe(self._update_x, names=["value"])
        self.text_y.observe(self._update, names=["value"])
        self.children = [self.text_x, self.text_y]
        # self.value = f"{x},{y}"

    def _update(self, change):
        print(change)
        if change["name"] == "x":
            self.text_x.value = str(change["new"])
        elif change["name"] == "y":
            self.text_y.value = str(change["new"])
        elif change["name"] == "text_x":
            self.x = change["new"]
        elif change["name"] == "text_y":
            self.y = change["new"]


_LETTERS_AND_DIGITS = string.ascii_lowercase[:6] + string.digits


def random_hex_color():
    return "#" + "".join(random.choices(_LETTERS_AND_DIGITS, k=6))


class LinearFace(traitlets.HasTraits):
    angle = traitlets.Float()
    callback = traitlets.Any()

    def __init__(self, x1=0, y1=0, x2=1, y2=1, height=10, width=2):
        self.p1 = Point(0, 0, description="p0")
        self.p2 = Point(0, 0, description="p1")
        self.height = widgets.Text(description="height", value=str(height))
        self.width = widgets.Text(description="Width", value=str(width))
        # self.midpoint = some_math.Point(*some_math.midpoint(self.p1, self.p2), description="midpoint")
        # self.rect_origin = some_math.Point(0,0, description="rect origin")
        self.midpoint = Point(0, 0, description="midpoint", disabled=True)
        self.rect_origin = Point(0, 0, description="rect origin", disabled=True)
        self.text_angle = widgets.Text("", description="rot (deg)")
        self.color_picker = widgets.ColorPicker(
            description="line", value=random_hex_color()
        )
        self.name = widgets.Text(description="name", value=word_pair(), disabled=True)
        self.offset = widgets.Text(description='offset', value='0')
        recompute = [self.p1, self.p2, self.height, self.width]
        self.redraw = [self.offset] + recompute + [self.color_picker]
        self.derived = [
            widgets.Label("Derived"),
            self.midpoint,
            self.rect_origin,
            self.text_angle,
        ]
        self.layout = widgets.HBox(
            [
                widgets.VBox([widgets.Label("Inputs")] + self.redraw + [self.name]),
                widgets.VBox(self.derived),
            ]
        )

        # wire up interactivity
        for widg in recompute:
            widg.observe(self._update_derived, names="value")
        self.offset.observe(self._update_points, names=['value'])
        self.observe(self._update_angle, names=["text_angle"])
        self.text_angle.observe(self._update_angle, names=["value"])

        self.p1.x, self.p1.y = x1, y1
        self.p2.x, self.p2.y = x2, y2

    def _update_points(self, change):
        offset = int(self.offset.value)
        orig_angle = angle(self.p1, self.p2)
        new_p1_x, new_p1_y = distance_polar(self.p1, offset, orig_angle + np.pi/2)
        new_p2_x, new_p2_y = distance_polar(self.p2, offset, orig_angle + np.pi/2)
        with self.hold_trait_notifications():
            self.p1.x = new_p1_x
            self.p1.y = new_p1_y
            self.p2.x = new_p2_x
            self.p2.y = new_p2_y
        self._update_derived({})
        

    def _update_derived(self, change):
        print(change)
        self.midpoint.x, self.midpoint.y = midpoint(self.p1, self.p2)
        # origin of rectangle should be "upstream" of the midpoint, so we reverse the point vector
        r, th = to_polar(self.p2, self.p1)
        self.rect_angle = th * 180 / np.pi
        x, y = distance_polar(self.midpoint, float(self.width.value), th)
        self.rect_origin.x, self.rect_origin.y = x, y
        if self.callback is not None:
            self.callback(self)

    def _update_angle(self, change):
        if change["name"] == "value":
            self.angle = float(self.rect_angle.value)
        elif change["name"] == "angle":
            self.rect_angle.value = str(self.angle)
        else:
            print(f"not sure what to do with {change}")


#     def _update_rectangle(self, change):
#         x,y = self._compute_midpoint()
#         width = float(self.width.value)
#         length = float(self.length.value)
#         p1 =


## point functions
def to_polar(p1: Point, p2: Point) -> Tuple[float, float]:
    """Compute r and theta from p1, p2"""
    r = distance(p1, p2)
    theta = angle(p1, p2)
    return r, theta


def distance(p1: Point, p2: Point) -> float:
    return spatial.distance.euclidean((p1.x, p1.y), (p2.x, p2.y))


def angle(p1: Point, p2: Point) -> float:
    th = np.arctan2((p2.y - p1.y), (p2.x - p1.x))
    return th


def angle_degrees(p1: Point, p2: Point) -> float:
    a = angle(p1, p2)
    return a * 180 / np.pi


def midpoint(p1: Point, p2: Point) -> Tuple[float, float]:
    xm = (p1.x + p2.x) / 2
    ym = (p1.y + p2.y) / 2
    return (xm, ym)


def distance_polar(p: Point, r: float, th: float) -> Tuple[float, float]:
    x = r * np.cos(th) + p.x
    y = r * np.sin(th) + p.y
    return x, y


def mpl_rect_patch(axes, x, y, width, height, rot, color=None):
    """Helper function to generate a rectangle patch

    Parameters
    ----------
    axes: mpl axes object
    `x` coord in data space
    `y` coord in data space
    `width` of rectangle
    `height` of rectangle
    `rot`. angle, in degrees, of rotation around (x,y)
    `color` is the color for the patch

    Returns
    -------
    patch: generated patch
    """
    if color is None:
        color = random_hex_color()
    patch = patches.Rectangle(
        xy=(x, y), width=width, height=height, color=color, alpha=0.5
    )
    t2 = mpl.transforms.Affine2D().rotate_deg_around(x, y, rot) + axes.transData
    patch.set_transform(t2)
    return patch


def imshow_with_patches(ax, data, patches: dict):
    try:
        im = ax.images[-1]
    except IndexError:
        print("No image currently show. calling ax.imshow")
        im = ax.imshow(data)
    else:
        im.set_data(data)

    print(f"Removing {len(ax.patches)} existing patches")

    for existing_patch in ax.patches:
        existing_patch.remove()
    added_patches = {}
    for patch_name, patch in patches.items():
        print(f"Adding patch for {patch_name}")
        added_patches[patch_name] = ax.add_patch(patch)
    return im, added_patches


def make_coords_array(arr):
    x = np.arange(0, arr.shape[1], step=1)
    y = np.arange(0, arr.shape[0], step=1)
    x_arr = np.zeros(shape=arr.shape, dtype=int)
    y_arr = np.zeros(shape=arr.shape, dtype=int)
    x_arr[:] = x
    y_arr.T[:] = y
    coords = np.zeros(shape=(x_arr.ravel().shape[0], 2), dtype=int)
    coords[:, 0] = x_arr.ravel()
    coords[:, 1] = y_arr.ravel()
    return coords


def zero_outside_patch(image_data, patch):
    coords = make_coords_array(image_data)
    transformed_coords = patch.axes.transData.transform(coords)
    within_patch = patch.contains_points(transformed_coords)
    # reshape the boolean mask to be the same shape as the image data
    within_patch.shape = image_data.shape
    im2 = image_data.copy()
    im2[~within_patch] = 0
    return im2


def make_image_mask(image_data, patch):
    coords = make_coords_array(image_data)
    transformed_coords = patch.axes.transData.transform(coords)
    within_patch = patch.contains_points(transformed_coords)
    # reshape the boolean mask to be the same shape as the image data
    within_patch.shape = image_data.shape
    return within_patch


def extract_pixels_xyi(image_data, patch):
    mask = make_image_mask(image_data, patch)
    I = image_data[mask]
    coords = make_coords_array(image_data)
    X = coords[..., 0].reshape(image_data.shape)[mask]
    Y = coords[..., 1].reshape(image_data.shape)[mask]
    return X, Y, I
