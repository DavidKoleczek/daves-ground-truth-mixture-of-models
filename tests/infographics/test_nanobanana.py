"""Test for Gemini CLI nanobanana image generation extension."""

import os
from pathlib import Path
import tempfile

import pytest

from model_diversity.gemini import query_gemini

_OUTPUT_SUBDIR = "nanobanana-output"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
async def test_nanobanana_generate_image() -> None:
    """Verify the nanobanana extension generates an image file via the Gemini CLI."""
    with tempfile.TemporaryDirectory(prefix="nanobanana_test_") as tmpdir:
        result = await query_gemini(
            prompt=(
                "Use the generate_image tool to generate a single image of a cat sitting on a windowsill. "
                "Do not open a preview."
            ),
            model="gemini-3.1-pro-preview",
            cwd=tmpdir,
            approval_mode="yolo",
        )

        output_dir = Path(tmpdir) / _OUTPUT_SUBDIR
        assert output_dir.exists(), f"Expected nanobanana output directory at {output_dir}"

        image_files = list(output_dir.iterdir())
        assert len(image_files) >= 1, f"Expected at least one generated image, got: {image_files}"

        image_path = image_files[0]
        assert image_path.stat().st_size > 1000, f"Image file too small ({image_path.stat().st_size} bytes)"

        # Copy to a stable location so the user can inspect the result
        stable_dir = Path("/tmp/hle_nanobanana_test")
        stable_dir.mkdir(parents=True, exist_ok=True)
        stable_path = stable_dir / image_path.name
        stable_path.write_bytes(image_path.read_bytes())
        print(f"\nGenerated image saved to: {stable_path}")
        print(f"Gemini response: {result.text}")
