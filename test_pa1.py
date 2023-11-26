import pytest
import numpy as np
import scipy.sparse as sp
import pa1


def test_discreteSecondDiff():
    mat = pa1.discreteSecondDiff(4).todense()
    assert np.array_equal(
        mat,
        np.array([
            [-2,  1,  0,  0],
            [ 1, -2,  1,  0],
            [ 0,  1, -2,  1],
            [ 0,  0,  1, -2]])) # yapf: disable


def test_vectorizedLaplace2x3():
    lap = pa1.vectorizedLaplace(2, 3).todense()
    assert np.array_equal(
        lap,
        np.array([
            [-4,  1,  1,  0,  0,  0],
            [ 1, -4,  0,  1,  0,  0],
            [ 1,  0, -4,  1,  1,  0],
            [ 0,  1,  1, -4,  0,  1],
            [ 0,  0,  1,  0, -4,  1],
            [ 0,  0,  0,  1,  1, -4]])) # yapf: disable

def test_vectorizedLaplace3x3():
    lap = pa1.vectorizedLaplace(3, 3).todense()
    assert np.array_equal(
        lap,
        np.array([
            [-4,  1,  0,  1,  0,  0,  0,  0,  0],
            [ 1, -4,  1,  0,  1,  0,  0,  0,  0],
            [ 0,  1, -4,  0,  0,  1,  0,  0,  0],
            [ 1,  0,  0, -4,  1,  0,  1,  0,  0],
            [ 0,  1,  0,  1, -4,  1,  0,  1,  0],
            [ 0,  0,  1,  0,  1, -4,  0,  0,  1],
            [ 0,  0,  0,  1,  0,  0, -4,  1,  0],
            [ 0,  0,  0,  0,  1,  0,  1, -4,  1],
            [ 0,  0,  0,  0,  0,  1,  0,  1, -4]])) # yapf: disable
