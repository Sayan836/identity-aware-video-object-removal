import unittest

from logo_removal.vlm_analysis import (
    VLM_PROVIDER_HEURISTIC,
    VLM_PROVIDER_NONE,
    VLM_PROVIDER_OLLAMA,
    VlmAnalysisResult,
    build_affected_mask_provider,
    generate_void_prompt,
    validate_vlm_provider,
)


class VlmAnalysisTests(unittest.TestCase):
    def test_crowded_human_prompt_preserves_neighboring_people(self):
        result = generate_void_prompt(
            base_prompt="clean natural background after the selected object is removed",
            removal_mode="ai_human_crowded",
            vlm_provider=VLM_PROVIDER_HEURISTIC,
        )

        self.assertEqual(result.strategy, "crowded_human_preservation")
        self.assertIn("Remove only the selected target person", result.prompt)
        self.assertIn("Preserve every other person", result.prompt)
        self.assertIn("contact, shadow", result.prompt)

    def test_static_prompt_remains_manual(self):
        result = generate_void_prompt(
            base_prompt="remove marked area",
            removal_mode="static_rectangle",
            vlm_provider=VLM_PROVIDER_NONE,
        )

        self.assertEqual(result.prompt, "remove marked area")
        self.assertEqual(result.strategy, "manual_static")

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "none, ollama"):
            validate_vlm_provider("gemini")

    def test_ollama_prompt_uses_scene_guidance(self):
        analysis = VlmAnalysisResult(
            provider=VLM_PROVIDER_OLLAMA,
            model="llama3.2-vision:11b",
            prompt_hint="restore empty gray pavement and the soft contact shadow",
            confidence=0.8,
        )

        result = generate_void_prompt(
            base_prompt="clean natural background after the selected object is removed",
            removal_mode="ai_human_crowded",
            vlm_provider=VLM_PROVIDER_OLLAMA,
            vlm_analysis=analysis,
        )

        self.assertEqual(result.strategy, "crowded_human_preservation_ollama_scene")
        self.assertIn("Ollama VLM", result.prompt)
        self.assertIn("gray pavement", result.prompt)

    def test_ollama_provider_applies_supplied_analysis_without_network(self):
        class DummyMetadata:
            width = 8
            height = 8

        class DummyPrimaryProvider:
            pass

        class DummyCv2:
            MORPH_ELLIPSE = 0

            def getStructuringElement(self, *_args):
                return None

        analysis = VlmAnalysisResult(
            provider=VLM_PROVIDER_OLLAMA,
            model="llava:7b",
            contact_dilation_px=22,
            shadow_dilation_px=44,
            shadow_vertical_offset_px=18,
            shadow_horizontal_offset_px=-12,
            confidence=0.9,
        )
        provider = build_affected_mask_provider(
            vlm_provider=VLM_PROVIDER_OLLAMA,
            np=object(),
            cv2=DummyCv2(),
            metadata=DummyMetadata(),
            primary_mask_provider=DummyPrimaryProvider(),
            removal_mode="ai_human_crowded",
            vlm_analysis=analysis,
        )

        provider.prepare()

        self.assertEqual(provider.contact_dilation_px, 22)
        self.assertEqual(provider.shadow_dilation_px, 44)
        self.assertEqual(provider.shadow_vertical_offset_px, 18)
        self.assertEqual(provider.shadow_horizontal_offset_px, -12)
        self.assertEqual(provider.provider_name, VLM_PROVIDER_OLLAMA)


if __name__ == "__main__":
    unittest.main()
