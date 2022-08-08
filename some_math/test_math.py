from . import LinearFace, Point, distance, angle_degrees, to_polar, midpoint, distance_polar
import numpy as np
import pytest


def test_distance():
    x0,y0 = 0,0
    x1,y1 = 1,1
    p0 = Point(x0,y0)
    p1 = Point(x1,y1)

    np_dist = np.sqrt((p1.x-p0.x)**2 + (p1.y-p0.y)**2)
    assert distance(p0, p1) == np_dist


def test_angle():
    p0 = Point(0, 0)
    p1 = Point(1, 1)
    p2 = Point(-1, 1)
    p3 = Point(-1, -1)
    p4 = Point(1, -1)

    assert angle_degrees(p0, p1) == pytest.approx(45)
    assert angle_degrees(p0, p2) == pytest.approx(135)
    assert angle_degrees(p0, p3) == pytest.approx(-135)
    assert angle_degrees(p0, p4) == pytest.approx(-45)

def test_to_polar():
    p0 = Point(0,0)
    p1 = Point(1,1)
    r, th = to_polar(p0, p1)
    assert r == np.sqrt(2)
    assert th == np.pi / 4


def test_widget():
    p0 = Point(0,0)
    p0.x = 1
    
    assert p0.value == "1.0,0.0"

    p0.value = "2,3"
    assert p0.x == pytest.approx(2.0)
    assert p0.y == pytest.approx(3.0)


def test_midpoint():
    p0 = Point(0,0)
    p1 = Point(4,2)

    x1,y1 = midpoint(p0, p1)
    assert x1 == pytest.approx(2)
    assert y1 == pytest.approx(1)

    x2,y2 = midpoint(p1, p0)
    assert x2 == pytest.approx(2)
    assert y2 == pytest.approx(1)


def test_distance_polar():
    p0 = Point(0, 0)
    r = np.sqrt(2)
    th = np.pi / 4

    x, y = distance_polar(p0, np.sqrt(2), np.pi/4)
    assert x == pytest.approx(1)
    assert y == pytest.approx(1)

    x, y = distance_polar(p0, np.sqrt(2), np.pi*3/4)
    assert x == pytest.approx(-1)
    assert y == pytest.approx(1)

    x, y = distance_polar(p0, r, -np.pi/4)
    assert x == pytest.approx(1)
    assert y == pytest.approx(-1)

    x, y = distance_polar(p0, r, -np.pi*3/4)
    assert x == pytest.approx(-1)
    assert y == pytest.approx(-1)


def test_line_thing():
    x0, y0 = 0,0
    x1,y1 = 2,4
    width = np.sqrt((x1/2)**2 + (y1/2)**2)
    length = 10
    lf = LinearFace(x0, y0, x1, y1, length=length, width=width)

    assert lf.midpoint.x == pytest.approx(x1/2)
    assert lf.midpoint.y == pytest.approx(y1/2)
    assert lf.rect_origin.x == pytest.approx(0)
    assert lf.rect_origin.y == pytest.approx(0)