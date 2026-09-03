"""
Fetcher security tests (plan §23.2, §34.2).

/v1/process accepts URLs and fetches them from inside a Pod on a third-party
network. Without restriction it is an SSRF primitive for anyone who gets past the
bearer token. These tests pin down the restrictions.

The most important assertion in this file is the first one: an EMPTY allowlist
rejects everything. Failing closed means a misconfigured worker is a visible
outage; failing open would be an invisible hole.
"""

from __future__ import annotations

import pytest

from app.fetcher import UrlRejected, _host_allowed, redact, validate_url

ALLOWED = ["s3.us-west-004.backblazeb2.com"]


class TestHostMatching:
    """The pure allowlist matcher, tested without DNS.

    Separated from validate_url on purpose: validate_url also resolves the host
    to reject private addresses, so testing subdomain logic through it would make
    these assertions depend on the test machine's DNS.
    """

    def test_exact_match(self):
        assert _host_allowed("s3.us-west-004.backblazeb2.com", ALLOWED)

    def test_subdomain_of_an_allowed_host_matches(self):
        assert _host_allowed("sub.backblazeb2.com", ["backblazeb2.com"])
        assert _host_allowed("a.b.backblazeb2.com", ["backblazeb2.com"])

    def test_suffix_confusion_does_not_match(self):
        """`notbackblazeb2.com` must not satisfy an allowlist of
        `backblazeb2.com`. The match is anchored on a leading dot precisely to
        stop this."""
        assert not _host_allowed("notbackblazeb2.com", ["backblazeb2.com"])
        assert not _host_allowed("backblazeb2.com.evil.com", ["backblazeb2.com"])

    def test_unrelated_host_does_not_match(self):
        assert not _host_allowed("evil.example.com", ALLOWED)

    def test_empty_allowlist_matches_nothing(self):
        assert not _host_allowed("s3.us-west-004.backblazeb2.com", [])

    def test_blank_entries_are_ignored(self):
        """A trailing comma in ALLOWED_URL_HOSTS must not become a wildcard."""
        assert not _host_allowed("evil.example.com", ["", "  "])


class TestAllowlist:
    """End-to-end validate_url, using hosts that genuinely resolve."""

    def test_empty_allowlist_rejects_everything(self):
        """Fail closed. This is the single most important behaviour here."""
        with pytest.raises(UrlRejected, match="no allowed hosts"):
            validate_url("https://s3.us-west-004.backblazeb2.com/b/k.jpg", [])

    def test_exact_host_is_allowed(self):
        host = validate_url(
            "https://s3.us-west-004.backblazeb2.com/bucket/large/ab/cd.jpg?X-Amz-Signature=x",
            ALLOWED,
        )
        assert host == "s3.us-west-004.backblazeb2.com"

    def test_unrelated_host_is_rejected(self):
        with pytest.raises(UrlRejected, match="not in the allowlist"):
            validate_url("https://evil.example.com/x.jpg", ALLOWED)

    def test_host_matching_is_case_insensitive(self):
        assert validate_url("https://S3.US-WEST-004.BACKBLAZEB2.COM/x.jpg", ALLOWED)

    def test_unresolvable_host_is_rejected_even_when_allowlisted(self):
        """An allowlisted name that does not resolve is still refused.

        That is the correct direction: we cannot verify it is not internal, so we
        do not fetch it.
        """
        with pytest.raises(UrlRejected, match="could not be resolved"):
            validate_url(
                "https://does-not-exist.backblazeb2.com/x.jpg", ["backblazeb2.com"]
            )


class TestScheme:
    def test_http_is_rejected(self):
        with pytest.raises(UrlRejected, match="not https"):
            validate_url("http://s3.us-west-004.backblazeb2.com/x.jpg", ALLOWED)

    def test_file_scheme_is_rejected(self):
        with pytest.raises(UrlRejected):
            validate_url("file:///etc/passwd", ALLOWED)

    def test_gopher_scheme_is_rejected(self):
        with pytest.raises(UrlRejected):
            validate_url("gopher://127.0.0.1:6379/_FLUSHALL", ALLOWED)


class TestCredentialsAndShape:
    def test_embedded_credentials_are_rejected(self):
        """A URL carrying `user:pass@` is either a mistake or an attempt to reach
        something that needs them."""
        with pytest.raises(UrlRejected, match="credentials"):
            validate_url(
                "https://user:pass@s3.us-west-004.backblazeb2.com/x.jpg", ALLOWED
            )

    def test_empty_url_is_rejected(self):
        with pytest.raises(UrlRejected):
            validate_url("", ALLOWED)

    def test_absurdly_long_url_is_rejected(self):
        with pytest.raises(UrlRejected):
            validate_url(
                "https://s3.us-west-004.backblazeb2.com/" + "a" * 3000, ALLOWED
            )

    def test_missing_host_is_rejected(self):
        with pytest.raises(UrlRejected):
            validate_url("https:///just/a/path", ALLOWED)


class TestPrivateAddressGuard:
    """Defence in depth behind the allowlist.

    Also catches DNS rebinding: an allowlisted NAME repointed at an internal
    address.
    """

    def test_loopback_hostname_is_rejected_even_if_allowlisted(self):
        with pytest.raises(UrlRejected, match="non-public|not in the allowlist"):
            validate_url("https://localhost/x.jpg", ["localhost"])

    def test_literal_loopback_ip_is_rejected(self):
        with pytest.raises(UrlRejected, match="non-public|not in the allowlist"):
            validate_url("https://127.0.0.1/x.jpg", ["127.0.0.1"])

    def test_link_local_metadata_address_is_rejected(self):
        """169.254.169.254 is the cloud metadata endpoint — the canonical SSRF
        target."""
        with pytest.raises(UrlRejected, match="non-public|not in the allowlist"):
            validate_url(
                "https://169.254.169.254/latest/meta-data/", ["169.254.169.254"]
            )

    def test_private_range_is_rejected(self):
        with pytest.raises(UrlRejected, match="non-public|not in the allowlist"):
            validate_url("https://10.0.0.5/x.jpg", ["10.0.0.5"])


class TestRedaction:
    def test_query_string_is_stripped(self):
        """A presigned URL's query string contains the SigV4 signature, which is a
        bearer credential for the object. It must never reach a log."""
        url = (
            "https://s3.us-west-004.backblazeb2.com/bucket/large/ab/cd.jpg"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeefcafe"
        )
        redacted = redact(url)
        assert "deadbeefcafe" not in redacted
        assert "X-Amz-Signature" not in redacted
        assert "s3.us-west-004.backblazeb2.com" in redacted
        assert "/bucket/large/ab/cd.jpg" in redacted

    def test_unparseable_input_does_not_raise(self):
        assert redact("\x00\x01not a url") is not None
