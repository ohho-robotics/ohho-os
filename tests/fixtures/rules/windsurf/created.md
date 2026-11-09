<!-- ohho-agent-rules:created -->
<!-- ohho-agent-rules:start v1 -->
# Ohho agent rules

Follow https://ohho-robotics.com/onboard.md.

- Use `sim://` first and by default for every connection, drive, and check.
- Never arm hardware. Never set `OHHO_ARM_HARDWARE`. Never type the arming confirmation. Never pass `--arm`.
- Never call `release-stop` or `release_stop`. Never change safety limits or the safety config.
- Emergency stop is always allowed. Calling `emergency_stop` is allowed.
- On any SafetyGate or other safety refusal, stop and tell the user. Do not retry around it.
- Ask the human for anything physical: power, placing the robot, clearing the area, and arming. The human arms hardware in their own terminal.
- Report only state you have verified. If a step did not run, say so.
- Check setup with `ohho doctor --agent`. Check the simulator with `ohho verify --sim`.
- Use the `ohho mcp serve` tools `connect`, `drive`, `telemetry`, `nav_goto`, and `look` with transport `sim://`. Do not point those tools at a hardware URI.
<!-- ohho-agent-rules:end -->
