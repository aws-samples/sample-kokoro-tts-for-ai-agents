"""L3 construct for a vLLM-backed SageMaker endpoint with bidirectional streaming."""

from __future__ import annotations

import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import aws_cdk.aws_sagemaker as sagemaker
from constructs import Construct

from speech_infra.config import ModelEndpointConfig


class VllmStreamingEndpoint(Construct):
    """SageMaker real-time endpoint using vLLM with WebSocket streaming support."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_config: ModelEndpointConfig,
        execution_role: iam.IRole,
        image_uri: str,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        endpoint_name = model_config.endpoint_name
        environment = env_overrides if env_overrides else dict(model_config.container_env)

        self.model = sagemaker.CfnModel(
            self,
            "Model",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                environment=environment,
            ),
        )

        self.endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "EndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="primary",
                    model_name=self.model.attr_model_name,
                    instance_type=model_config.instance_type,
                    initial_instance_count=max(model_config.min_instances, 1),
                    container_startup_health_check_timeout_in_seconds=(
                        model_config.container_startup_health_check_timeout_s
                    ),
                    routing_config=sagemaker.CfnEndpointConfig.RoutingConfigProperty(
                        routing_strategy="LEAST_OUTSTANDING_REQUESTS",
                    ),
                )
            ],
        )
        self.endpoint_config.add_dependency(self.model)

        self.endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_name=endpoint_name,
            endpoint_config_name=self.endpoint_config.attr_endpoint_config_name,
            # Without this, CloudFormation keeps asserting the variant properties it
            # synthesized — including initial_instance_count — while autoscaling is
            # busy changing DesiredInstanceCount. The two fight, and a deploy can
            # silently scale a busy endpoint back down to its initial count. That
            # race is why autoscaling was disabled outright in 6997191; retaining
            # the live values is the actual fix.
            retain_all_variant_properties=True,
        )
        self.endpoint.add_dependency(self.endpoint_config)

        cdk.CfnOutput(
            scope,
            f"{endpoint_name}-EndpointName",
            value=endpoint_name,
        )
