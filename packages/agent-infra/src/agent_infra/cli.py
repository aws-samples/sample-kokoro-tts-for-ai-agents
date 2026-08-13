"""CLI for deploying and managing the agent-infra demo, and running its latency demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

CDK_APP_DIR = Path(__file__).resolve().parent.parent.parent

STACK_NAMES = ["AgentFoundation", "AgentRuntime"]


@click.group()
def cli() -> None:
    """Manage the agent-infra demo's AgentCore Runtime deployment."""


@cli.command()
@click.option("--bedrock-model-id", required=True, help="Bedrock model ID for the Strands agent.")
@click.option("--image-uri", default=None, help="Full container image URI (skips Docker build).")
def diff(bedrock_model_id: str, image_uri: str | None) -> None:
    """Show what would change for the agent stacks."""
    context = {"agent:bedrock_model_id": bedrock_model_id}
    if image_uri:
        context["image_uri"] = image_uri
    _run_cdk("diff", STACK_NAMES, context=context)


@cli.command()
@click.option("--bedrock-model-id", required=True, help="Bedrock model ID for the Strands agent.")
@click.option("--image-uri", default=None, help="Full container image URI (skips Docker build).")
def deploy(bedrock_model_id: str, image_uri: str | None) -> None:
    """Deploy the agent's AgentCore Runtime."""
    context = {"agent:bedrock_model_id": bedrock_model_id}
    if image_uri:
        context["image_uri"] = image_uri
    _run_cdk("deploy", STACK_NAMES, context=context, require_approval="never")


@cli.command()
@click.option("--bedrock-model-id", required=True, help="Bedrock model ID for the Strands agent.")
def destroy(bedrock_model_id: str) -> None:
    """Tear down the agent's AgentCore Runtime."""
    _run_cdk(
        "destroy", STACK_NAMES, context={"agent:bedrock_model_id": bedrock_model_id}, force=True
    )


@cli.command()
def status() -> None:
    """Show deployment status via CloudFormation."""
    for stack_name in STACK_NAMES:
        click.echo(f"{stack_name}: checking...")


@cli.command()
@click.option("--endpoint-url", default="http://localhost:8080", help="Agent server base URL.")
@click.option(
    "--prompt",
    default="Tell me a short story about a robot learning to sing.",
    help="Prompt sent to the agent for both modes.",
)
@click.option("--voice", default="af_heart", help="Kokoro voice ID to synthesize with.")
def demo(endpoint_url: str, prompt: str, voice: str) -> None:
    """Run the bidi-vs-batch latency comparison against a running agent server."""
    from agent_infra.scripts.compare_latency_demo import run_comparison

    run_comparison(endpoint_url, prompt, voice)


def _run_cdk(
    command: str,
    stacks: list[str],
    context: dict[str, str] | None = None,
    require_approval: str | None = None,
    force: bool = False,
) -> None:
    """Execute a CDK command via subprocess."""
    cmd = ["cdk", command, *stacks, "--app", "python -m agent_infra.app"]

    if context:
        for key, value in context.items():
            cmd.extend(["-c", f"{key}={value}"])

    if require_approval:
        cmd.extend(["--require-approval", require_approval])

    if force:
        cmd.append("--force")

    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=CDK_APP_DIR)  # noqa: S603
    if result.returncode != 0:
        sys.exit(result.returncode)
