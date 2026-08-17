# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""SigV4 credential resolution for the bidirectional (HTTP/2) transport.

Ported verbatim from ``tts_bench.bidi.Boto3CredentialsResolver``.
"""

from __future__ import annotations

from typing import Any

import boto3
from smithy_aws_core.identity import AWSCredentialsIdentity

from tts_client.errors import TTSClientError


class Boto3CredentialsResolver:
    """Resolve SigV4 credentials through boto3's full credential chain.

    The bidi SDK's own ``EnvironmentCredentialsResolver`` reads
    ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` directly and raises
    when they are unset. On an EC2 instance with an IAM role they *are*
    unset, so that resolver cannot authenticate at all. The other bundled
    resolvers do not substitute cleanly either: ``IMDSCredentialsResolver``
    requires an ``http_client`` argument, and ``StaticCredentialsResolver``
    reads from auth properties rather than config.

    Deferring to boto3 also means the bidi transport and the response-stream
    transport authenticate identically, so a permissions difference between
    transports cannot masquerade as a capacity or availability difference.

    ``get_frozen_credentials()`` is called on **every** resolve rather than
    cached. A long-running bidi session can outlive role credentials that
    rotate mid-session; caching the first frozen tuple (as the SDK's own
    resolver does) would start failing partway through, at exactly the point
    a wave of auth failures is most easily misread as something else.
    """

    def __init__(self, session: Any | None = None) -> None:
        self._session = session if session is not None else boto3.Session()

    async def get_identity(self, *, properties: Any = None) -> AWSCredentialsIdentity:
        """Return current credentials as an ``AWSCredentialsIdentity``."""
        credentials = self._session.get_credentials()
        if credentials is None:
            raise TTSClientError(
                "no AWS credentials found by boto3's credential chain; bidi streaming "
                "cannot be signed. Check the instance role or AWS_PROFILE."
            )
        frozen = credentials.get_frozen_credentials()
        return AWSCredentialsIdentity(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=frozen.token,
        )
