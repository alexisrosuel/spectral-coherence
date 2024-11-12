import mlx.core as mx


def mx_real(x: mx.array) -> mx.array:
    return x.astype(mx.float32)


def mx_imag(x: mx.array) -> mx.array:
    return -(1j * x).astype(mx.float32)


def mx_conj(x):
    return mx_real(x) - 1j * mx_imag(x)


def mx_matmul(a: mx.array, b: mx.array) -> mx.array:
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
    ar, ai = mx_real(a), mx_imag(a)
    br, bi = mx_real(b), mx_imag(b)

    real_part = mx.einsum(subscripts, ar, br) - mx.einsum(subscripts, ai, bi)
    imag_part = mx.einsum(subscripts, ar, bi) + mx.einsum(subscripts, ai, br)

    return real_part + 1j * imag_part
