# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for Boto3CredentialsResolver.

Ported from tts_bench/tests/test_bidi.py's TestBoto3CredentialsResolver,
which tested the pre-tts-client copy of this exact class.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tts_client.bidi_credentials import Boto3CredentialsResolver
from tts_client.errors import TTSClientError


class _FrozenCreds:
    def __init__(self, key: str, secret: str, token: str | None) -> None:
        self.access_key = key
        self.secret_key = secret
        self.token = token


class _RotatingCredentials:
    """Mimics botocore's refreshable credentials: a new tuple per freeze."""

    def __init__(self) -> None:
        self.freezes = 0

    def get_frozen_credentials(self) -> _FrozenCreds:
        self.freezes += 1
        return _FrozenCreds(f"AKIA{self.freezes}", f"secret{self.freezes}", f"token{self.freezes}")


class TestBoto3CredentialsResolver:
    def test_resolves_through_boto3s_chain(self) -> None:
        session = MagicMock()
        session.get_credentials.return_value = _RotatingCredentials()
        identity = asyncio.run(Boto3CredentialsResolver(session=session).get_identity())

        assert identity.access_key_id == "AKIA1"
        assert identity.secret_access_key == "secret1"
        # The session token is what makes role credentials work at all; dropping
        # it produces a signature the service rejects.
        assert identity.session_token == "token1"

    def test_refreezes_on_every_resolve_rather_than_caching(self) -> None:
        # This is the whole reason the class exists. A long-running bidi
        # session can outlive role credentials that rotate mid-session; a
        # cached tuple would start failing partway through, at exactly the
        # point a wave of auth failures is most easily misread as something
        # else.
        credentials = _RotatingCredentials()
        session = MagicMock()
        session.get_credentials.return_value = credentials
        resolver = Boto3CredentialsResolver(session=session)

        first = asyncio.run(resolver.get_identity())
        second = asyncio.run(resolver.get_identity())

        assert credentials.freezes == 2
        assert first.access_key_id != second.access_key_id

    def test_re_reads_the_session_each_time_so_a_new_role_is_picked_up(self) -> None:
        session = MagicMock()
        session.get_credentials.return_value = _RotatingCredentials()
        resolver = Boto3CredentialsResolver(session=session)

        asyncio.run(resolver.get_identity())
        asyncio.run(resolver.get_identity())

        assert session.get_credentials.call_count == 2

    def test_missing_credentials_raise_rather_than_silently_fail(self) -> None:
        # An unauthenticated call is a setup fault; it must raise rather than
        # be classified as some capacity outcome by a caller.
        session = MagicMock()
        session.get_credentials.return_value = None
        with pytest.raises(TTSClientError, match="no AWS credentials"):
            asyncio.run(Boto3CredentialsResolver(session=session).get_identity())

    def test_default_session_is_boto3_session_when_none_given(self) -> None:
        resolver = Boto3CredentialsResolver()
        import boto3

        assert isinstance(resolver._session, boto3.Session)
