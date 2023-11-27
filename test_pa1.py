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


def test_vectorizedLaplace_2x3():
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

def test_vectorizedLaplace_3x3():
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

def test_getSystem_boundary():
    target = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]) # yapf:disable
    source = np.zeros((3, 3))
    (A, b) = pa1.getSystem(source, target, 0, 0)
    A = A.todense()
    ident = np.eye(9)
    boundaryIndices = [x for x in range(9) if x != 4]
    assert np.array_equal(ident[boundaryIndices, :], A[boundaryIndices, :])
    assert np.array_equal([1, 4, 7, 2, 8, 3, 6, 9], b[boundaryIndices])


def test_getSystem_gradient():
    target = np.zeros((3, 3))
    source = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]) # yapf:disable
    (A, b) = pa1.getSystem(source, target, 0, 0)
    assert b[4] == -4 * 5 + 2 + 4 + 6 + 8


def test_clone():
    target = np.zeros((3, 3))
    source = np.array([
        [1, 2, 3],
        [4, 60, 6],
        [10, 15, 20]]) # yapf:disable
    result = pa1.clone(source, target, 0, 0)
    assert result[1, 1] == round((-4 * 60 + 2 + 4 + 6 + 15) / -4.)
    assert np.count_nonzero(result) == 1
