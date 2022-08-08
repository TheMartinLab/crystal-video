
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

WORDS = []
compounds = ['CZX-1', 'ZnCl2', 'CuCl', 'halozeotype', 'amine', 'hydrate']

def word_pair():
    global WORDS
    if not WORDS:
        p = pathlib.Path(__file__).absolute().parent.parent / 'english-adjectives.txt'
        WORDS = open(p, 'r').read().split('\n')
    return f'{random.choice(WORDS)} {random.choice(compounds)}'

class Point(widgets.Text):
    x = traitlets.Float()
    y = traitlets.Float()
    
    def __init__(self, x, y, *args, **kwargs):
        super(Point, self).__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.observe(self._update, names=['x', 'y', 'value'])
        self.value = f'{x},{y}'
        
        
    def _update(self, change):
        print(change)
        if change['name'] == 'value':
            x,y = self.value.split(',')
            self.x = float(x)
            self.y = float(y)
        if change['name'] == 'x' or change['name'] == 'y':
            self.value = f'{self.x},{self.y}'

_LETTERS_AND_DIGITS = string.ascii_lowercase[:6] + string.digits
def random_hex_color():
    return '#' + ''.join(random.choices(_LETTERS_AND_DIGITS, k=6))

class LinearFace(traitlets.HasTraits):
    rect_angle = traitlets.Float()
    callback = traitlets.Any()
    
    def __init__(self, x1=0, y1=0, x2=1, y2=1, length=10, width=2):
        self.p1 = Point(0,0, description="p0")
        self.p2 = Point(0,0, description="p1")
        self.length = widgets.Text(description="Length", value=str(length))
        self.width = widgets.Text(description="Width", value=str(width))
        # self.midpoint = some_math.Point(*some_math.midpoint(self.p1, self.p2), description="midpoint")
        # self.rect_origin = some_math.Point(0,0, description="rect origin")
        self.midpoint = Point(0,0, description="midpoint", disabled=True)
        self.rect_origin = Point(0,0, description="rect origin", disabled=True)
        self.widg_rect_angle = widgets.Text("", description="rot (deg)", disabled=True)
        self.color_picker = widgets.ColorPicker(
            description='line',
            value=random_hex_color()
        )
        self.name = widgets.Text(description="name", value=word_pair(), disabled=True)
        
        recompute = [self.p1, self.p2, self.length, self.width]
        self.redraw = recompute + [self.color_picker]
        self.derived = [widgets.Label("Derived"), self.midpoint, self.rect_origin, self.widg_rect_angle]
        self.layout = widgets.HBox([
            widgets.VBox([widgets.Label("Inputs")] + self.redraw + [self.name]),
            widgets.VBox(self.derived),
        ])
        
        # wire up interactivity
        for widg in recompute:
            widg.observe(self._update_derived, names='value')
            
        self.observe(self._update_rect_angle, names=['rect_angle'])
            
        self.p1.x, self.p1.y = x1, y1
        self.p2.x, self.p2.y = x2, y2
        
    def _update_derived(self,change):
        print(change)
        self.midpoint.x, self.midpoint.y = midpoint(self.p1, self.p2)
        # origin of rectangle should be "upstream" of the midpoint, so we reverse the point vector
        r, th = to_polar(self.p2, self.p1)
        self.rect_angle = th * 180 / np.pi
        x, y = distance_polar(self.midpoint, float(self.width.value), th)
        self.rect_origin.x, self.rect_origin.y = x,y
        if self.callback is not None:
            self.callback(self)
    
    def _update_rect_angle(self, change):
        self.widg_rect_angle.value = f'{self.rect_angle}'
    
        
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
    return spatial.distance.euclidean(
        (p1.x, p1.y), (p2.x, p2.y))

def angle(p1: Point, p2: Point) -> float:
    th = np.arctan2((p2.y-p1.y), (p2.x-p1.x))
    return th

def angle_degrees(p1: Point, p2: Point) -> float:
    a = angle(p1, p2)
    return a * 180 / np.pi
    

def midpoint(p1: Point, p2: Point) -> Tuple[float, float]: 
    xm = (p1.x + p2.x) / 2
    ym = (p1.y + p2.y) / 2
    return (xm,ym)


def distance_polar(p: Point, r: float, th: float) -> Tuple[float, float]:
    x = r * np.cos(th) + p.x
    y = r * np.sin(th) + p.y
    return x, y