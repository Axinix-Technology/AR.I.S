# pyright: ignore
"""
AR.I.S — Dynamic Capability Auto-Resolution & Self-Knowledge Cache Tests

Verifies:
1. Capabilities auto-resolve dynamically from ArisCapability subclasses without static dictionaries.
2. The persistent capability knowledge cache (data/aris_capabilities_cache.json) is automatically synced.
3. AR.I.S knows its capabilities and operational usage via get_capability_usage().
4. Dynamic capabilities registered at runtime are discovered and cached automatically.
"""

import os
import json
import unittest

from api.semantic.executable_tools import (
    ArisCapability,
    ArisCapabilityRegistry,
    ArisMathTool,
    ArisBrowserAccessTool,
    ArisCommandPromptTool,
)


class CapabilityAutoResolveAndSelfKnowledgeTests(unittest.TestCase):
    """
    Validates dynamic capability discovery, self-knowledge cache, and usage semantics.
    """

    def test_dynamic_auto_resolve_discovers_all_core_capabilities(self):
        """Verifies that auto_resolve discovers math, browser, and command_prompt tools."""
        caps = ArisCapabilityRegistry.auto_resolve(sync_cache=True)
        self.assertIn("math_computation", caps)
        self.assertIn("browser_access", caps)
        self.assertIn("command_prompt", caps)

        self.assertIs(ArisCapabilityRegistry.get_capability("math_computation"), ArisMathTool)
        self.assertIs(ArisCapabilityRegistry.get_capability("browser_access"), ArisBrowserAccessTool)
        self.assertIs(ArisCapabilityRegistry.get_capability("command_prompt"), ArisCommandPromptTool)

    def test_capabilities_cache_file_persists_full_specifications(self):
        """Verifies that data/aris_capabilities_cache.json exists and contains full metadata & usage."""
        cache_path = ArisCapabilityRegistry.CACHE_FILE_PATH
        self.assertTrue(os.path.exists(cache_path), f"Cache file must exist at {cache_path}")

        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        for cap_name in ["math_computation", "browser_access", "command_prompt"]:
            self.assertIn(cap_name, cache_data, f"Cache must contain specification for '{cap_name}'")
            spec = cache_data[cap_name]
            self.assertTrue(len(spec.get("usage", "")) > 10, f"Usage guidance must be populated for '{cap_name}'")
            self.assertTrue(bool(spec.get("domain")), f"Domain must be populated for '{cap_name}'")
            self.assertIn("entrypoint", spec, f"Entrypoint must be recorded for '{cap_name}'")

    def test_aris_self_knowledge_retrieves_operational_usage(self):
        """Verifies AR.I.S can query and retrieve operational usage semantics for any capability."""
        math_usage = ArisCapabilityRegistry.get_capability_usage("math_computation")
        self.assertIsNotNone(math_usage)
        self.assertIn("arithmetic", math_usage.lower())

        browser_usage = ArisCapabilityRegistry.get_capability_usage("browser_access")
        self.assertIsNotNone(browser_usage)
        self.assertIn("web", browser_usage.lower())

        cmd_usage = ArisCapabilityRegistry.get_capability_usage("command_prompt")
        self.assertIsNotNone(cmd_usage)
        self.assertIn("shell", cmd_usage.lower())

        # All capabilities usage map
        all_usage = ArisCapabilityRegistry.get_all_capabilities_usage()
        self.assertGreaterEqual(len(all_usage), 3)
        self.assertIn("command_prompt", all_usage)

    def test_dynamic_runtime_subclass_is_auto_discovered_and_cached(self):
        """Verifies a brand-new ArisCapability subclass is auto-discovered and added to cache."""
        class MockAutonomousSensorTool(ArisCapability):
            CAPABILITY_NAME = "sensor_monitor"
            ALIASES = ["sensor", "telemetry"]
            IS_EXTERNAL = False
            AFFORDANCE_DOMAIN = "hardware_telemetry"
            DESCRIPTION = "Monitors hardware temperatures and load telemetry."
            USAGE = "Use sensor_monitor when system CPU or GPU telemetry is requested."
            PARAMETERS_SCHEMA = {"sensor_id": "Identifier of the sensor to query"}
            ACCEPTS = {"domain": "hardware_telemetry", "requires": ["sensor_id"]}
            PRODUCES = {"output_type": "telemetry_reading"}

            @classmethod
            def execute(cls, **kwargs):
                return {"status": "success", "reading": 42.0}

        try:
            # Auto-resolve should discover MockAutonomousSensorTool via ArisCapability.__subclasses__()
            resolved = ArisCapabilityRegistry.auto_resolve(sync_cache=True)
            self.assertIn("sensor_monitor", resolved)
            self.assertIs(ArisCapabilityRegistry.get_capability("sensor_monitor"), MockAutonomousSensorTool)
            self.assertIs(ArisCapabilityRegistry.get_capability("telemetry"), MockAutonomousSensorTool)

            # Verify usage is accessible
            usage = ArisCapabilityRegistry.get_capability_usage("sensor_monitor")
            self.assertIn("telemetry", usage.lower())

            # Verify written to cache
            cache = ArisCapabilityRegistry.sync_cache()
            self.assertIn("sensor_monitor", cache)
            self.assertEqual(cache["sensor_monitor"]["domain"], "hardware_telemetry")
        finally:
            # Clean up test tool from memory
            ArisCapabilityRegistry._CAPABILITIES.pop("sensor_monitor", None)


if __name__ == "__main__":
    unittest.main()
