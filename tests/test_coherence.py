import numpy as np
import pytest

# from spectral_coherence.coherence import, coherence


# @pytest.mark.parametrize(
#     "X, expected_X",
#     [
#         ([[1]], [[1]]),
#         ([[1, 2], [3, 4]], [[1.0, 1.0], [1.5, 1.0]]),
#     ],
# )
# def test__normalize(X, expected_X):
#     # note that _normalize expect to receive a 3D array, where the
#     # first dimension is the number of samples (or Fourier frequencies)
#     X, expected_X = np.array([X]), np.array([expected_X])
#     np.testing.assert_allclose(_normalize(X), expected_X)


# @pytest.mark.parametrize(
#     "x, B, expected_coherence, expected_freqs",
#     [
#         # 1 feature, so the coherence is always 1
#         (
#             [[1], [2], [3], [4], [5]],
#             1,
#             [
#                 [[1.0 + 0.0j]],
#                 [[1.0 + 0.0j]],
#                 [[1.0 + 0.0j]],
#                 [[1.0 + 0.0j]],
#                 [[1.0 + 0.0j]],
#             ],
#             [-0.4, -0.2, 0.0, 0.2, 0.4],
#         ),
#         # 2 features
#         (
#             [[1, 2], [2, -3], [3, 4]],
#             1,
#             [
#                 [
#                     [1.0 + 0.0j, 0.2773501 - 0.96076892j],
#                     [0.2773501 + 0.96076892j, 1.0 + 0.0j],
#                 ],
#                 [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
#                 [
#                     [1.0 + 0.0j, 0.2773501 + 0.96076892j],
#                     [0.2773501 - 0.96076892j, 1.0 + 0.0j],
#                 ],
#             ],
#             [-0.33333333, 0.0, 0.33333333],
#         ),
#     ],
# )
# def test_coherence(x, B, expected_coherence, expected_freqs):
#     x, expected_coherence, expected_freqs = (
#         np.array(x),
#         np.array(expected_coherence),
#         np.array(expected_freqs),
#     )
#     Cs, freqs = coherence(x, B)
#     assert np.allclose(Cs, expected_coherence)
#     assert np.allclose(freqs, expected_freqs)
