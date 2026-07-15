import torch

from aurora.normalisation import normalise_surf_var
from utils.transform_data import transform_data


def test_transform_data_roundtrip_tp_mswep_tensor() -> None:
    x = torch.tensor([0.0, 1e-6, 1e-4, 2e-3], dtype=torch.float32)
    x_t = transform_data(x, "tp_mswep", direct=True)
    x_back = transform_data(x_t, "tp_mswep", direct=False)
    torch.testing.assert_close(x_back, x, rtol=1e-5, atol=1e-8)


def test_normalise_tp_mswep_uses_transformed_values() -> None:
    x = torch.tensor([0.0, 1e-5, 1e-4], dtype=torch.float32)
    locations = {"tp_mswep": 0.0}
    scales = {"tp_mswep": 1.0}

    x_norm = normalise_surf_var(x, "tp_mswep", locations, scales)
    expected = transform_data(x, "tp_mswep", direct=True)

    torch.testing.assert_close(x_norm, expected, rtol=1e-6, atol=1e-8)


def test_normalise_unnormalise_roundtrip_tp_mswep() -> None:
    x = torch.tensor([0.0, 1e-6, 2e-5, 6e-4], dtype=torch.float32)
    locations = {"tp_mswep": 0.7}
    scales = {"tp_mswep": 2.3}

    x_norm = normalise_surf_var(x, "tp_mswep", locations, scales)
    x_back = normalise_surf_var(x_norm, "tp_mswep", locations, scales, unnormalise=True)

    torch.testing.assert_close(x_back, x, rtol=1e-5, atol=1e-8)
