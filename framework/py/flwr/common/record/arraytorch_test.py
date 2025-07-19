# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for Array Torch functions."""

import importlib.util
import unittest

from .array import Array

# Check if torch is installed
torch_spec = importlib.util.find_spec("torch")
torch_available = torch_spec is not None

# Set flags only if torch is available
bfloat16_capable = False
if torch_available:
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        bfloat16_capable = major >= 8  # Ampere+ GPUs (A100, RTX 30xx, etc.)
    elif hasattr(torch.backends, "cpu") and hasattr(
        torch.backends.cpu, "has_bf16_support"
    ):
        bfloat16_capable = torch.backends.cpu.has_bf16_support()


# Skip test class entirely if torch or bfloat16 is unavailable
@unittest.skipUnless(torch_available, "PyTorch not available")
@unittest.skipUnless(bfloat16_capable, "System is not bfloat16 capable")
class TestArrayTorch(unittest.TestCase):
    """Tests for Array Torch functionality."""

    def test_bfloat16_tensor_round_trip(self):
        """Test compression and decompression of bfloat16 tensor."""
        # Create a bfloat16 PyTorch tensor
        original_tensor = torch.randn(3, 3, dtype=torch.bfloat16)

        # Convert to Array using torch_tensor constructor
        arr = Array(torch_tensor=original_tensor)

        self.assertEqual(arr.dtype, str(original_tensor.dtype))
        self.assertEqual(arr.shape, tuple(original_tensor.shape))
        self.assertEqual(arr.stype, "safetensor")
        self.assertIsInstance(arr.data, bytes)

        recovered_tensor = arr.torch()
        self.assertEqual(recovered_tensor.shape, original_tensor.shape)
        self.assertEqual(recovered_tensor.dtype, torch.bfloat16)

        import numpy as np

        np.testing.assert_allclose(
            recovered_tensor.float().numpy(),
            original_tensor.float().numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_torch_raises_for_invalid_stype(self):
        """Verify invalid stypes are raised"""
        from your_module.array import Array  # Replace with your actual path

        arr = Array(
            dtype="float32", shape=(2, 2), stype="invalid_stype", data=b"somebytes"
        )
        with self.assertRaises(TypeError):
            arr.torch()


    def test_from_torch_tensor_with_torch(self) -> None:
        """Test creating an Array from a real PyTorch tensor."""

        # Prepare tensor
        tensor = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

        # Execute
        arr = Array.from_torch_tensor(tensor)

        # Deserialize to verify
        loaded = arr.torch()

        # Assert
        self.assertEqual(arr.dtype, "torch.float32")
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.stype, SType.SAFETENSOR)
        torch.testing.assert_close(loaded, tensor)

