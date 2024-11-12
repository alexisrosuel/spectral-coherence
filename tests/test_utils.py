import mlx.core as mx
import pytest
from spectral_coherence.utils import mx_conj, mx_einsum, mx_imag, mx_matmul, mx_real


@pytest.mark.parametrize(
    "x, expected",
    [
        (mx.array([1 + 1j, 2 + 2j]), mx.array([1 - 1j, 2 - 2j])),
    ],
)
def test_mx_conj(x, expected):
    result = mx_conj(x)
    assert mx.array_equal(result, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (mx.array([1 + 1j, 2 + 2j]), mx.array([1.0, 2.0])),
    ],
)
def test_mx_real(x, expected):
    result = mx_real(x)
    assert mx.array_equal(result, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (mx.array([1 + 1j, 2 + 2j]), mx.array([1.0, 2.0])),
    ],
)
def test_mx_imag(x, expected):
    result = mx_imag(x)
    assert mx.array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            mx.array([[1.0, 2.0], [3.0, 4.0]]),
            mx.array([[5.0, 6.0], [7.0, 8.0]]),
            mx.array([[19.0, 22.0], [43.0, 50.0]]),
        ),
        (
            mx.array([[1.0j, 2.0j], [3.0j, 4.0j]]),
            mx.array([[5.0j, 6.0j], [7.0j, 8.0j]]),
            mx.array([[-19.0, -22.0], [-43.0, -50.0]]),
        ),
    ],
)
def test_mx_matmul(a, b, expected):
    result = mx_matmul(a, b)
    assert mx.array_equal(result, expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (
            mx.array([[1, 2], [3, 4]]),
            mx.array([[5, 6], [7, 8]]),
            mx.array([[19, 22], [43, 50]]),
        ),
    ],
)
def test_mx_einsum(a, b, expected):
    result = mx_einsum("ij,jk->ik", a, b)
    assert mx.array_equal(result, expected)
