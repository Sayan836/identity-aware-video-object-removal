import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from logo_removal.void_pipeline import (
    QUALITY_RESTORATION_FFMPEG_BICUBIC,
    QUALITY_RESTORATION_REALESRGAN,
    QUALITY_RESTORATION_VENHANCER,
    QUALITY_RESTORATION_NONE,
    VoidRuntimeConfig,
    build_void_pass1_command,
    post_enhance_void_output,
    resolve_void_resource_profile,
    save_stable_chunk_output,
    stitch_void_chunk_outputs,
    validate_void_runtime_config,
    validate_temporal_window_size,
)


class VoidPipelineTests(unittest.TestCase):
    def test_l4_pro_balanced_matches_notebook_config(self):
        profile = resolve_void_resource_profile("l4_pro_balanced")

        self.assertEqual(profile.sample_size, "256x448")
        self.assertEqual(profile.max_video_length, 45)
        self.assertEqual(profile.temporal_window_size, 45)
        self.assertEqual(profile.latent_temporal_frames, 12)
        self.assertEqual(profile.gpu_memory_mode, "model_cpu_offload_and_qfloat8")
        self.assertEqual(profile.num_inference_steps, 20)
        self.assertEqual(profile.temporal_multidiffusion_stride, 16)

    def test_temporal_window_rejects_odd_latent_count(self):
        with self.assertRaisesRegex(ValueError, "latent frame count"):
            validate_temporal_window_size(49)

    def test_void_command_uses_notebook_flags(self):
        command = build_void_pass1_command(
            save_path=Path("/content/out"),
            data_root=Path("/content/data"),
            sequence_name="phase5_custom",
            runtime_config=VoidRuntimeConfig(
                void_repo=Path("/content/void-model"),
                resource_profile="l4_pro_balanced",
            ),
            package_fps=29.833333,
        )

        self.assertIn("--config", command)
        self.assertIn("config/quadmask_cogvideox.py", command)
        self.assertIn("--config.data.sample_size=256x448", command)
        self.assertIn("--config.data.max_video_length=45", command)
        self.assertIn("--config.data.fps=30", command)
        self.assertIn("--config.video_model.temporal_window_size=45", command)
        self.assertIn("--config.video_model.num_inference_steps=20", command)
        self.assertIn("--config.system.gpu_memory_mode=model_cpu_offload_and_qfloat8", command)
        self.assertIn("--config.system.device=cuda", command)

    def test_void_runtime_validation_reports_missing_repo_before_chunking(self):
        with self.assertRaisesRegex(FileNotFoundError, "VOID runtime is not ready"):
            validate_void_runtime_config(
                VoidRuntimeConfig(
                    void_repo=Path("/definitely/missing/void-model"),
                    resource_profile="l4_pro_balanced",
                )
            )

    def test_stable_chunk_output_retimes_to_source_fps_and_size(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch("logo_removal.void_pipeline.subprocess.run") as run,
            patch("logo_removal.void_pipeline.probe_video") as probe,
        ):
            tmp = Path(tmpdir)
            probe.return_value.width = 640
            probe.return_value.height = 360
            probe.return_value.fps = 29.97002997
            probe.return_value.frame_count = 45
            probe.return_value.duration = 45 / 29.97002997

            save_stable_chunk_output(
                source_path=tmp / "void_raw.mp4",
                target_path=tmp / "void_chunk_000.mp4",
                frame_count=45,
                fps=29.97002997,
                width=640,
                height=360,
            )

        extract_command = run.call_args_list[0].args[0]
        assemble_command = run.call_args_list[1].args[0]
        self.assertIn("-frames:v", extract_command)
        self.assertIn("45", extract_command)
        self.assertIn("-framerate", assemble_command)
        self.assertIn("29.97003", assemble_command)
        self.assertIn("-vf", assemble_command)
        self.assertIn("setpts=PTS-STARTPTS,scale=640:360:flags=lanczos", assemble_command)
        self.assertIn("-crf", assemble_command)
        self.assertIn("16", assemble_command)

    def test_stitch_output_caps_total_frame_count(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch("logo_removal.void_pipeline.subprocess.run") as run,
            patch("logo_removal.void_pipeline.probe_video") as probe,
        ):
            tmp = Path(tmpdir)
            chunk_a = tmp / "void_chunk_000.mp4"
            chunk_b = tmp / "void_chunk_001.mp4"
            probe.return_value.width = 640
            probe.return_value.height = 360
            probe.return_value.fps = 30.0
            probe.return_value.frame_count = 90
            probe.return_value.duration = 3.0

            stitch_void_chunk_outputs(
                chunk_outputs=[chunk_a, chunk_b],
                output_path=tmp / "merged.mp4",
                fps=30.0,
                width=640,
                height=360,
                frame_count=90,
            )

        command = run.call_args.args[0]
        self.assertIn("-f", command)
        self.assertIn("concat", command)
        self.assertIn("-c", command)
        self.assertIn("copy", command)

    def test_post_enhance_none_returns_original_path(self):
        path = Path("/tmp/void_chunk.mp4")

        result = post_enhance_void_output(
            input_path=path,
            quality_restoration=QUALITY_RESTORATION_NONE,
            fps=30.0,
            width=640,
            height=360,
            frame_count=45,
        )

        self.assertEqual(result, path)

    def test_post_enhance_ffmpeg_bicubic_normalizes_timing(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch("logo_removal.void_pipeline.subprocess.run") as run,
            patch("logo_removal.void_pipeline.probe_video") as probe,
        ):
            tmp = Path(tmpdir)
            source = tmp / "merged.mp4"
            source.write_bytes(b"placeholder")
            probe.return_value.width = 640
            probe.return_value.height = 360
            probe.return_value.fps = 30.0
            probe.return_value.frame_count = 90
            probe.return_value.duration = 3.0

            result = post_enhance_void_output(
                input_path=source,
                quality_restoration=QUALITY_RESTORATION_FFMPEG_BICUBIC,
                fps=30.0,
                width=640,
                height=360,
                frame_count=90,
            )

        self.assertEqual(result.name, "merged_restored.mp4")
        command = run.call_args.args[0]
        self.assertIn("-vf", command)
        self.assertIn("setpts=PTS-STARTPTS,scale=640:360:flags=bicubic,fps=30", command)
        self.assertIn("-frames:v", command)
        self.assertIn("90", command)

    def test_venhancer_uses_only_venhancer_template(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(
                "os.environ",
                {
                    "VENHANCER_COMMAND_TEMPLATE": "python venhance.py --input {input} --output {output}",
                    "REALESRGAN_COMMAND_TEMPLATE": "python restore.py --input {input} --output {output}",
                },
                clear=True,
            ),
            patch("logo_removal.void_pipeline._run_streaming_shell_command") as run,
            patch("logo_removal.void_pipeline.validate_timed_video_output"),
        ):
            run.return_value = (0, "")
            source = Path(tmpdir) / "merged.mp4"
            source.write_bytes(b"placeholder")

            result = post_enhance_void_output(
                input_path=source,
                quality_restoration=QUALITY_RESTORATION_VENHANCER,
                fps=30.0,
                width=640,
                height=360,
                frame_count=90,
            )

        self.assertEqual(result.name, "merged_venhancer.mp4")
        self.assertIn("venhance.py", run.call_args.args[0])

    def test_realesrgan_uses_only_realesrgan_template(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(
                "os.environ",
                {
                    "VENHANCER_COMMAND_TEMPLATE": "python venhance.py --input {input} --output {output}",
                    "REALESRGAN_COMMAND_TEMPLATE": "python restore.py --input {input} --output {output}",
                },
                clear=True,
            ),
            patch("logo_removal.void_pipeline._run_streaming_shell_command") as run,
            patch("logo_removal.void_pipeline.validate_timed_video_output"),
        ):
            run.return_value = (0, "")
            source = Path(tmpdir) / "merged.mp4"
            source.write_bytes(b"placeholder")

            result = post_enhance_void_output(
                input_path=source,
                quality_restoration=QUALITY_RESTORATION_REALESRGAN,
                fps=30.0,
                width=640,
                height=360,
                frame_count=90,
            )

        self.assertEqual(result.name, "merged_realesrgan.mp4")
        self.assertIn("restore.py", run.call_args.args[0])

    def test_selected_external_restoration_requires_matching_template(self):
        with TemporaryDirectory() as tmpdir, patch.dict("os.environ", {}, clear=True):
            source = Path(tmpdir) / "merged.mp4"
            source.write_bytes(b"placeholder")

            with self.assertRaisesRegex(RuntimeError, "REALESRGAN_COMMAND_TEMPLATE"):
                post_enhance_void_output(
                    input_path=source,
                    quality_restoration=QUALITY_RESTORATION_REALESRGAN,
                    fps=30.0,
                    width=640,
                    height=360,
                    frame_count=90,
                )

    def test_external_restoration_failure_logs_command_output(self):
        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(
                "os.environ",
                {
                    "REALESRGAN_COMMAND_TEMPLATE": "python restore.py --input {input} --output {output}",
                },
                clear=True,
            ),
            patch("logo_removal.void_pipeline._run_streaming_shell_command") as run,
        ):
            source = Path(tmpdir) / "merged.mp4"
            source.write_bytes(b"placeholder")
            logs = []

            def fail_command(command, log_callback, label):
                log_callback(f"{label}: wrapper stdout")
                log_callback(f"{label}: inner realesrgan traceback")
                return 7, "wrapper stdout\ninner realesrgan traceback"

            run.side_effect = fail_command

            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                post_enhance_void_output(
                    input_path=source,
                    quality_restoration=QUALITY_RESTORATION_REALESRGAN,
                    fps=30.0,
                    width=640,
                    height=360,
                    frame_count=90,
                    log_callback=logs.append,
                )

        self.assertTrue(any("wrapper stdout" in message for message in logs))
        self.assertTrue(any("inner realesrgan traceback" in message for message in logs))


if __name__ == "__main__":
    unittest.main()
