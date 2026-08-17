# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""CLI for deploying and managing speech model SageMaker endpoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from speech_infra.config import (
    TTS_MODEL_CONFIGS,
    ModelEndpointConfig,
    get_model_config,
)

CDK_APP_DIR = Path(__file__).resolve().parent.parent.parent


@click.group()
def cli() -> None:
    """Manage speech model SageMaker endpoints."""


@cli.command()
@click.argument("models", nargs=-1)
@click.option("--all", "diff_all", is_flag=True, help="Diff all configured models.")
@click.option("--image-uri", default=None, help="Full container image URI (skips Docker build).")
@click.option(
    "--hf-token-secret",
    default="hf-token-dev",
    help="Secrets Manager secret name for HuggingFace token.",
)
def diff(
    models: tuple[str, ...],
    diff_all: bool,
    image_uri: str | None,
    hf_token_secret: str,
) -> None:
    """Show what would change for one or more model endpoints."""
    targets = _resolve_targets(models, diff_all)
    if not targets:
        click.echo("No targets to diff.", err=True)
        sys.exit(1)

    stack_names = ["SpeechFoundation", "SpeechModelCache"]
    stack_names += [config.stack_id for config in targets]

    context: dict[str, str] = {"hf_token_secret": hf_token_secret}
    if image_uri:
        context["image_uri"] = image_uri

    click.echo(f"Diffing stacks: {', '.join(stack_names)}")
    _run_cdk("diff", stack_names, context=context)


@cli.command()
@click.argument("models", nargs=-1)
@click.option("--all", "deploy_all", is_flag=True, help="Deploy all configured models.")
@click.option("--image-uri", default=None, help="Full container image URI (skips Docker build).")
@click.option(
    "--hf-token-secret",
    default="hf-token-dev",
    help="Secrets Manager secret name for HuggingFace token.",
)
def deploy(
    models: tuple[str, ...],
    deploy_all: bool,
    image_uri: str | None,
    hf_token_secret: str,
) -> None:
    """Deploy one or more model endpoints."""
    targets = _resolve_targets(models, deploy_all)
    if not targets:
        click.echo("No targets to deploy.", err=True)
        sys.exit(1)

    stack_names = ["SpeechFoundation", "SpeechModelCache"]
    stack_names += [config.stack_id for config in targets]

    context: dict[str, str] = {"hf_token_secret": hf_token_secret}
    if image_uri:
        context["image_uri"] = image_uri

    click.echo(f"Deploying stacks: {', '.join(stack_names)}")
    _run_cdk("deploy", stack_names, context=context, require_approval="never")


@cli.command()
@click.argument("models", nargs=-1)
@click.option("--all", "destroy_all", is_flag=True, help="Destroy all endpoints.")
def destroy(models: tuple[str, ...], destroy_all: bool) -> None:
    """Tear down one or more model endpoints."""
    targets = _resolve_targets(models, destroy_all)
    if not targets:
        click.echo("No targets to destroy.", err=True)
        sys.exit(1)

    stack_names = [config.stack_id for config in targets]

    click.echo(f"Destroying stacks: {', '.join(stack_names)}")
    _run_cdk("destroy", stack_names, force=True)


@cli.command("list")
def list_models() -> None:
    """List all configured models and their endpoint names."""
    click.echo(f"{'Model':<20} {'Endpoint':<28} {'Instance':<16} {'Container'}")
    click.echo("-" * 80)
    for name, config in TTS_MODEL_CONFIGS.items():
        click.echo(
            f"{name:<20} {config.endpoint_name:<28} "
            f"{config.instance_type:<16} {config.container_type.value}"
        )


@cli.command()
@click.argument("model", required=False)
def status(model: str | None) -> None:
    """Show endpoint deployment status via CloudFormation."""
    if model:
        configs = [get_model_config(model)]
    else:
        configs = list(TTS_MODEL_CONFIGS.values())

    for config in configs:
        click.echo(f"{config.stack_id}: checking...")


def _resolve_targets(
    models: tuple[str, ...],
    all_models: bool,
) -> list[ModelEndpointConfig]:
    """Resolve CLI arguments into model configs."""
    if all_models:
        return list(TTS_MODEL_CONFIGS.values())

    model_configs: list[ModelEndpointConfig] = []
    for m in models:
        try:
            model_configs.append(get_model_config(m))
        except KeyError as e:
            click.echo(str(e), err=True)
            sys.exit(1)

    return model_configs


def _ecr_login(account: str, region: str) -> None:
    """Authenticate docker to an ECR registry."""
    login_cmd = (
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin "
        f"{account}.dkr.ecr.{region}.amazonaws.com"
    )
    result = subprocess.run(login_cmd, shell=True, capture_output=True)  # noqa: S602
    if result.returncode != 0:
        click.echo(f"ECR login failed for account {account}.", err=True)
        sys.exit(1)


def _run_cdk(
    command: str,
    stacks: list[str],
    context: dict[str, str] | None = None,
    require_approval: str | None = None,
    force: bool = False,
) -> None:
    """Execute a CDK command via subprocess."""
    cmd = ["cdk", command, *stacks, "--app", "python -m speech_infra.app"]

    if context:
        for key, value in context.items():
            cmd.extend(["-c", f"{key}={value}"])

    if require_approval:
        cmd.extend(["--require-approval", require_approval])

    if force:
        cmd.append("--force")

    cmd.append("--concurrency=10")

    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=CDK_APP_DIR)  # noqa: S603
    if result.returncode != 0:
        sys.exit(result.returncode)
