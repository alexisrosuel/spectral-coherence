import mlx.core as mx


def mx_real(x: mx.array) -> mx.array:
    """
    Returns the real part of the input. If the input is real, it is returned
    as is. If the input is complex, the real part is returned.
    """
    return x.astype(mx.float32)


def mx_imag(x: mx.array) -> mx.array:
    """
    Returns the imaginary part of the input. If the input is real, it is returned
    as is. If the input is complex, the imaginary part is returned.
    """
    return -(1j * x).astype(mx.float32)


def mx_conj(x):
    return mx_real(x) - 1j * mx_imag(x)


def mx_matmul(a: mx.array, b: mx.array) -> mx.array:
    """
    Perform complex matrix multiplication between two MLX arrays.

    This function handles the following cases:
    1. Both inputs are complex
    2. One input is complex and the other is real
    3. Both inputs are real (falls back to standard matmul)

    Parameters:
    -----------
    a : mx.array
        First input array
    b : mx.array
        Second input array

    Returns:
    --------
    result : mx.array
        Result of the complex matrix multiplication

    Note:
    -----
    This function assumes that the input arrays are compatible for matrix multiplication.
    It does not check for shape compatibility.
    """
    a_is_complex = a.dtype == mx.complex64
    b_is_complex = b.dtype == mx.complex64

    if a_is_complex and b_is_complex:
        # Both inputs are complex
        a_real, a_imag = mx_real(a), mx_imag(a)
        b_real, b_imag = mx_real(b), mx_imag(b)

        real_part = a_real @ b_real - a_imag @ b_imag
        imag_part = a_real @ b_imag + a_imag @ b_real

    elif a_is_complex:
        # Only 'a' is complex
        a_real, a_imag = mx_real(a), mx_imag(a)

        real_part = a_real @ b
        imag_part = a_imag @ b

    elif b_is_complex:
        # Only 'b' is complex
        b_real, b_imag = mx_real(b), mx_imag(b)

        real_part = a @ b_real
        imag_part = a @ b_imag

    else:
        # Both inputs are real
        return a @ b

    return real_part + 1j * imag_part


def mx_einsum(subscripts, a, b):
    """
    Perform einsum operation on complex-valued arrays using real-valued einsum.
    """
    ar, ai = mx_real(a), mx_imag(a)
    br, bi = mx_real(b), mx_imag(b)

    real_part = mx.einsum(subscripts, ar, br) - mx.einsum(subscripts, ai, bi)
    imag_part = mx.einsum(subscripts, ar, bi) + mx.einsum(subscripts, ai, br)

    return real_part + 1j * imag_part
