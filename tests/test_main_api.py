import unittest

from fastapi import HTTPException

from main import validate_job_options


class ApiValidationTests(unittest.TestCase):
    def test_valid_generalized_options(self):
        validate_job_options(
            removal_mode="ai_object",
            inpaint_engine="opencv",
            method="telea",
            reference_frame=0,
            chunk_size=30,
            radius=3.0,
            mask_padding=2,
        )

    def test_invalid_removal_mode(self):
        with self.assertRaises(HTTPException):
            validate_job_options(
                removal_mode="unknown",
                inpaint_engine="opencv",
                method="telea",
                reference_frame=0,
                chunk_size=30,
                radius=3.0,
                mask_padding=0,
            )

    def test_invalid_inpaint_engine(self):
        with self.assertRaises(HTTPException):
            validate_job_options(
                removal_mode="static_rectangle",
                inpaint_engine="unknown",
                method="telea",
                reference_frame=0,
                chunk_size=30,
                radius=3.0,
                mask_padding=0,
            )

    def test_invalid_reference_frame(self):
        with self.assertRaises(HTTPException):
            validate_job_options(
                removal_mode="static_rectangle",
                inpaint_engine="opencv",
                method="telea",
                reference_frame=-1,
                chunk_size=30,
                radius=3.0,
                mask_padding=0,
            )

    def test_invalid_quality_restoration(self):
        with self.assertRaises(HTTPException):
            validate_job_options(
                removal_mode="static_rectangle",
                inpaint_engine="opencv",
                method="telea",
                reference_frame=0,
                chunk_size=30,
                radius=3.0,
                mask_padding=0,
                quality_restoration="unknown",
            )

    def test_invalid_vlm_provider(self):
        with self.assertRaises(HTTPException):
            validate_job_options(
                removal_mode="static_rectangle",
                inpaint_engine="opencv",
                method="telea",
                reference_frame=0,
                chunk_size=30,
                radius=3.0,
                mask_padding=0,
                vlm_provider="gemini",
            )


if __name__ == "__main__":
    unittest.main()
